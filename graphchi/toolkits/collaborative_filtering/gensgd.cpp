/**
 * @file
 * @author  Danny Bickson
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Carnegie Mellon University]
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

 *
 * @section DESCRIPTION
 *
 * Implementation of the gensgd algorithm. A generalization of SGD algorithm when there are multiple features for each
 * rating, in the form 
 * [from] [to] [feature1] [feature2] [feature3] ... [featureN] [rating] 
 * (It is also possible to dynamically specify column numbers which are relevant)
 * Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th IEEE International Conference on Data Mining (ICDM 2010), Sydney, Australia.
 * Original implementation by Qiang Yan, Chinese Academy of Science.
 * note: this code version implements the SGD version of gensgd. In the original library there are also ALS and MCMC methods.
 * Also the treatment of features is richer in gensgd. The code here can serve for a quick evaluation but the user
 * is encouraged to try gensgd as well.
 */


#include <vector>
#include "common.hpp"
#include "eigen_wrapper.hpp"
#include "../parsers/common.hpp"
#include <omp.h>
#define MAX_FEATURES 256
#define FEATURE_WIDTH 38 //MAX NUMBER OF ALLOWED FEATURES IN TEXT FILE

double gensgd_rate1 = 1e-02;
double gensgd_rate2 = 1e-02;
double gensgd_rate3 = 1e-02;
double gensgd_rate4 = 1e-02;
double gensgd_rate5 = 1e-02;
double gensgd_mult_dec = 0.9;
double gensgd_regw = 1e-3;
double gensgd_regv = 1e-3;
double gensgd_reg0 = 1e-1;
bool debug = false;
std::string user_file; //optional file with user features
std::string item_file; //optional file with item features
std::string user_links; //optional file with user to user links
int limit_rating = 0;
size_t vertex_with_no_edges = 0;
int calc_error = 0;
int file_columns = 0;
std::vector<std::string> header_titles;
int has_header_titles = 0;
float cutoff = 0;

struct stats{
  float minval;
  float maxval;
  float meanval;
  stats(){
    minval = maxval = meanval = 0;
  }
};

struct feature_control{
  std::vector<double_map> node_id_maps;
  double_map val_map;
  int rehash_value;
  int last_item;
  std::vector<stats> stats_array;
  int feature_num;
  int node_features;
  int node_links;
  int total_features;
  std::vector<bool> feature_selection;
  const std::string default_feature_str;
  std::vector<int> offsets;
  bool hash_strings;
  int from_pos;
  int to_pos;
  int val_pos;

  feature_control(){
    rehash_value = 0;
    last_item = 0;
    total_features = 0;
    node_features = 0;
    feature_num = FEATURE_WIDTH;
    hash_strings = false;
    from_pos = 0;
    to_pos = 1;
    val_pos = -1;
    node_links = 0;
    feature_selection.resize(MAX_FEATURES+3);
  }
};

feature_control fc;

int num_feature_bins(){
  int sum = 0;
  if (fc.hash_strings){
    assert(2+fc.total_features+fc.node_features == (int)fc.node_id_maps.size());
    for (int i=2; i < 2+fc.total_features+fc.node_features; i++){
      sum+= fc.node_id_maps[i].string2nodeid.size();
    }
  }
  else {
    for (int i=0; i< fc.total_features; i++)
      sum += (int)ceil((fc.stats_array[i].maxval - fc.stats_array[i].minval) + 1);
  }
  if (fc.total_features > 0)
    assert(sum > 0);
  return sum;
}

int calc_feature_num(){
  return 2+fc.total_features+fc.last_item+fc.node_features;
}
void get_offsets(std::vector<int> & offsets){
  assert(offsets.size() >= 2);
  offsets[0] = 0;
  offsets[1] = M;
  if (offsets.size() >= 3)
    offsets[2] = M+N;
 if (fc.hash_strings){
   for (uint j=2; j< offsets.size()-1; j++){
         offsets[j+1] = offsets[j] + fc.node_id_maps[j].string2nodeid.size();
          logstream(LOG_DEBUG)<<"Offset " << j+1 << " is: " << offsets[j+1] << std::endl;
   }
  } else {
      for (uint j=2; j < offsets.size(); j++){
           offsets[j+1] = offsets[j] + (int)ceil((fc.stats_array[j-2].maxval-fc.stats_array[j-2].minval)+1);
          logstream(LOG_DEBUG)<<"Offset " << j+1 << " is: " << offsets[j+1] << std::endl;
      }
  }

}


bool is_user(vid_t id){ return id < M; }
bool is_item(vid_t id){ return id >= M && id < N; }
bool is_time(vid_t id){ return id >= M+N; }

vec errors_vec;
#define BIAS_POS -1
struct vertex_data {
  vec pvec;
  double bias;
  int last_item;
  sparse_vec features;
  sparse_vec links; //links to other users or items

  vertex_data() {
    bias = 0;
    last_item = 0;
  }
  void set_val(int index, float val){
    if (index == BIAS_POS)
      bias = val;
    else pvec[index] = val;
  }
  float get_val(int index){
    if (index== BIAS_POS)
      return bias;
    else return pvec[index];
  }


};

struct edge_data {
  float features[FEATURE_WIDTH];
  float weight;
  edge_data() { weight = 0; memset(features, 0, sizeof(float)*FEATURE_WIDTH); }

  edge_data(float weight, float * valarray): weight(weight) { memcpy(features, valarray, sizeof(float)*FEATURE_WIDTH); }
};



/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program. 
 */
typedef vertex_data VertexDataType;
typedef edge_data EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL; 
std::vector<vertex_data> latent_factors_inmem;


int calc_feature_node_array_size(uint node, uint item){
  assert(node <= M);
  assert(item <= N);
  assert(node < latent_factors_inmem.size());
  assert(fc.offsets[1]+item < latent_factors_inmem.size());
  return 2+fc.total_features+fc.last_item+nnz(latent_factors_inmem[node].features)+nnz(latent_factors_inmem[fc.offsets[1]+item].features);
}



/**
 * return a numeric node ID out of the string text read from file (training, validation or test)
 */
float get_node_id(char * pch, int pos, size_t i, bool read_only = false){
  assert(pch != NULL);
  assert(i >= 0);

  float ret;
  //read numeric id
  if (!fc.hash_strings){
    ret = (pos < 2 ? atoi(pch) : atof(pch)); 
    if (pos < 2)
      ret-=input_file_offset;
    if (pos == 0 && ret >= M)
      logstream(LOG_FATAL)<<"Row index larger than the matrix row size " << ret << " > " << M << " in line: " << i << std::endl;
    else if (pos == 1 && ret >= N)
      logstream(LOG_FATAL)<<"Col index larger than the matrix row size " << ret << " > " << N << " in line: " << i << std::endl;

  }
  //else read string id and assign numeric id
  else {
    uint id;
    assert(pos < (int)fc.node_id_maps.size());
    if (read_only){ // find if node was in map
      std::map<std::string,uint>::iterator it = fc.node_id_maps[pos].string2nodeid.find(pch);
      if (it != fc.node_id_maps[pos].string2nodeid.end()){
          ret = it->second;
          assert(ret < fc.node_id_maps[pos].string2nodeid.size());
       }
       else ret = -1;
    } 
    else { //else enter node into map (in case it did not exist) and return its position 
      assign_id(fc.node_id_maps[pos], id, pch);
      ret = id;
    }
  }

  if (!read_only)
    assert(ret != -1);
  return ret;
}

float get_value(char * pch, bool read_only){
  float ret;
  if (!fc.rehash_value){
    ret = atof(pch);
  }
  else {
    uint id;
    if (read_only){ // find if node was in map
      std::map<std::string,uint>::iterator it = fc.val_map.string2nodeid.find(pch);
      if (it != fc.val_map.string2nodeid.end()){
          ret = it->second;
       }
       else ret = -1;
    } 
    else { //else enter node into map (in case it did not exist) and return its position 
      assign_id(fc.val_map, id, pch);
      ret = id;
    }
 
  }    
  if (std::isnan(ret) || std::isinf(ret))
    logstream(LOG_FATAL)<<"Failed to read value" << std::endl;
  return ret;
}

/* Read and parse one input line from file */
bool read_line(FILE * f, const std::string filename, size_t i, uint & I, uint & J, float &val, std::vector<float>& valarray, int type){

  char * linebuf = NULL;
  size_t linesize = 0;
  char linebuf_debug[1024];

  int token = 0;
  int index = 0;
  int rc = getline(&linebuf, &linesize, f);
  if (rc == -1)
    logstream(LOG_FATAL)<<"Failed to get line: " << i << " in file: " << filename << std::endl;

  char * linebuf_to_free = linebuf;
  strncpy(linebuf_debug, linebuf, 1024);

  assert(file_columns >= 2);
  while (token < file_columns){
    /* READ FROM */
    if (token == fc.from_pos){
      char *pch = strsep(&linebuf,"\t,\r\n ");
      if (pch == NULL)
        logstream(LOG_FATAL)<<"Error reading line " << i << " [ " << linebuf_debug << " ] " << std::endl;
      I = (uint)get_node_id(pch, 0, i);
      token++;
    }
    else if (token == fc.to_pos){
      /* READ TO */
      char * pch = strsep(&linebuf, "\t,\r\n ");
      if (pch == NULL)
        logstream(LOG_FATAL)<<"Error reading line " << i << " [ " << linebuf_debug << " ] " << std::endl;
      J = (uint)get_node_id(pch, 1, i);
      token++;
    }
    else if (token == fc.val_pos){
      /* READ RATING */
      char * pch = strsep(&linebuf, "\t,\r\n ");
      if (pch == NULL)
        logstream(LOG_FATAL)<<"Error reading line " << i << " [ " << linebuf_debug << " ] " << std::endl;

      val = get_value(pch, type != TRAINING);
      token++;
    }
    else {
      /* READ FEATURES */
      char * pch = strsep(&linebuf, "\t,\r\n ");
      if (pch == NULL)
        logstream(LOG_FATAL)<<"Error reading line " << i << " feature " << token << " [ " << linebuf_debug << " ] " << std::endl;
      if (!fc.feature_selection[token]){
        token++;
        continue;
      }

      assert(index < (int)valarray.size());
      valarray[index] = get_node_id(pch, index+2, i); 
      if (std::isnan(valarray[index]))
        logstream(LOG_FATAL)<<"Error reading line " << i << " feature " << token << " [ " << linebuf_debug << " ] " << std::endl;

      //calc stats about ths feature
      if (type == TRAINING && !fc.hash_strings){
        fc.stats_array[index].minval = std::min(fc.stats_array[index].minval, valarray[index]);
        fc.stats_array[index].maxval = std::max(fc.stats_array[index].maxval, valarray[index]);
        fc.stats_array[index].meanval += valarray[index];
      }


      index++;
      token++;
    }
  }//end while
  free(linebuf_to_free);
  return true;
}//end read_line

/* compute an edge prediction based on input features */
float compute_prediction(
    const uint I, 
    const uint J, 
    const float val, 
    double & prediction, 
    float * valarray, 
    float (*prediction_func)(const vertex_data ** array, int arraysize, float rating, double & prediction, vec * psum), 
    vec * psum, 
    vertex_data **& node_array){

  assert(J >=0 && J <= N);
  assert(I>=0 && I <= M);


  /* COMPUTE PREDICTION */
  /* USER NODE **/
  int index = 0;
  int loc = 0;
  node_array[index] = &latent_factors_inmem[I+fc.offsets[index]];
  assert(node_array[index]->pvec[0] < 1e5);
  index++; loc++;
  /* 1) ITEM NODE */
  assert(J+fc.offsets[index] < latent_factors_inmem.size());
  node_array[index] = &latent_factors_inmem[J+fc.offsets[index]];
  assert(node_array[index]->pvec[0] < 1e5);
  index++; loc++;
   /* 2) FEATURES GIVEN IN RATING LINE */
  for (int j=0; j< fc.total_features; j++){
    uint pos = (uint)ceil(valarray[j]+fc.offsets[j+index]-fc.stats_array[j].minval);
    assert(pos >= 0 && pos < latent_factors_inmem.size());
    node_array[j+index] = & latent_factors_inmem[pos];
    assert(node_array[j+index]->pvec[0] < 1e5);
  }
  index+= fc.total_features;
  loc += fc.total_features;
  /* 3) USER FEATURES */
  int i = 0;
  FOR_ITERATOR(j, latent_factors_inmem[I+fc.offsets[0]].features){
    int pos;
    if (user_links != ""){
    pos = j.index();
    assert(pos < M);
    }
    else {
    pos = j.index()+fc.offsets[index];
    assert(pos < (int)fc.node_id_maps[0].string2nodeid.size());
    assert(pos >= 0 && pos < latent_factors_inmem.size());
    assert(pos >= (uint)fc.offsets[index]);
    }
        //logstream(LOG_INFO)<<"setting index " << i+index << " to: " << pos << std::endl;
    node_array[i+index] = & latent_factors_inmem[pos];
    assert(node_array[i+index]->pvec[0] < 1e5);
    i++;
  }
  assert(i == nnz(latent_factors_inmem[I+fc.offsets[0]].features));
  index+= nnz(latent_factors_inmem[I+fc.offsets[0]].features);
  loc+=1;
  /* 4) ITEM FEATURES */
  i=0;
  FOR_ITERATOR(j, latent_factors_inmem[J+fc.offsets[1]].features){
    uint pos = j.index()+fc.offsets[loc];
    assert(j.index() < (int)fc.node_id_maps[loc].string2nodeid.size());
    assert(pos >= 0 && pos < latent_factors_inmem.size());
    assert(pos >= (uint)fc.offsets[loc]);
    //logstream(LOG_INFO)<<"setting index " << i+index << " to: " << pos << std::endl;
    node_array[i+index] = & latent_factors_inmem[pos];
    assert(node_array[i+index]->pvec[0] < 1e5);
    i++;
  }
  assert(i == nnz(latent_factors_inmem[J+fc.offsets[1]].features));
  index+= nnz(latent_factors_inmem[J+fc.offsets[1]].features);
  loc+=1;
  if (fc.last_item){
    uint pos = latent_factors_inmem[I].last_item + fc.offsets[2+fc.total_features+fc.node_features];
    assert(pos < latent_factors_inmem.size());
    node_array[index] = &latent_factors_inmem[pos];
    assert(node_array[index]->pvec[0] < 1e5);
    index++;
    loc+=1;
  }
  assert(index == calc_feature_node_array_size(I,J));
  (*prediction_func)((const vertex_data**)node_array, calc_feature_node_array_size(I,J), val, prediction, psum);
  return pow(val - prediction,2);
} 

#include "io.hpp"
#include "../parsers/common.hpp"


/**
 * Create a bipartite graph from a matrix. Each row corresponds to vertex
 * with the same id as the row number (0-based), but vertices correponsing to columns
 * have id + num-rows.
 * Line format of the type
 * [user] [item] [feature1] [feature2] ... [featureN] [rating]
 */

/* Read input file, process it and save a binary representation for faster loading */
template <typename als_edge_type>
int convert_matrixmarket_N(std::string base_filename, bool square, feature_control & fc, int limit_rating = 0) {
  // Note, code based on: http://math.nist.gov/MatrixMarket/mmio/c/example_read.c
  FILE *f;
  size_t nz;
  /**
   * Create sharder object
   */
  int nshards;
/*
  if ((nshards = find_shards<als_edge_type>(base_filename, get_option_string("nshards", "auto")))) {
    logstream(LOG_INFO) << "File " << base_filename << " was already preprocessed, won't do it again. " << std::endl;
    FILE * inf = fopen((base_filename + ".gm").c_str(), "r");
    int rc = fscanf(inf,"%d\n%d\n%ld\n%d\n%lg\n",&M, &N, &L, &fc.total_features, &globalMean);
    if (rc != 5)
      logstream(LOG_FATAL)<<"Failed to read global mean from file: " << base_filename << ".gm" << std::endl;
    for (int i=0; i< fc.total_features; i++){
      int rc = fscanf(inf, "%g\n%g\n%g\n", &fc.stats_array[i].minval, &fc.stats_array[i].maxval, &fc.stats_array[i].meanval);
      if (rc != 3)
        logstream(LOG_FATAL)<<"Failed to read global mean from file: " << base_filename << ".gm" << std::endl;

    }
    logstream(LOG_INFO) << "Read matrix of size " << M << " x " << N << " globalMean: " << globalMean << std::endl;
    for (int i=0; i< fc.total_features; i++){
      logstream(LOG_INFO) << "Feature " << i << " min val: " << fc.stats_array[i].minval << " max val: " << fc.stats_array[i].maxval << "  mean val: " << fc.stats_array[i].meanval << std::endl;
    }
    fclose(inf);

    if (fc.hash_strings){
      for (int i=0; i< fc.total_features+2; i++){
        char filename[256];
        sprintf(filename, "%s.feature%d.map", base_filename.c_str(),i);
        load_map_from_txt_file<std::map<std::string,uint> >(fc.node_id_maps[i].string2nodeid, filename, 2);
        if (fc.node_id_maps[i].string2nodeid.size() == 0)
          logstream(LOG_FATAL)<<"Failed to read " << filename << " please remove all temp files and try again" << std::endl;
      }
    }
    return nshards;
  }   
*/
  sharder<als_edge_type> sharderobj(base_filename);
  sharderobj.start_preprocessing();

  detect_matrix_size(base_filename, f, M, N, nz);
  if (f == NULL)
    logstream(LOG_FATAL) << "Could not open file: " << base_filename << ", error: " << strerror(errno) << std::endl;

  if (has_header_titles){
    char * linebuf = NULL;
    size_t linesize;
    char linebuf_debug[1024];

    /* READ LINE */
    int rc = getline(&linebuf, &linesize, f);
    if (rc == -1)
      logstream(LOG_FATAL)<<"Error header line " << " [ " << linebuf_debug << " ] " << std::endl;
      
    strncpy(linebuf_debug, linebuf, 1024);


    /** READ [FROM] */
    char *pch = strtok(linebuf,"\t,\r; ");
    if (pch == NULL)
      logstream(LOG_FATAL)<<"Error header line " << " [ " << linebuf_debug << " ] " << std::endl;
      
    header_titles.push_back(pch);

    /** READ USER FEATURES */
    while (pch != NULL){
      pch = strtok(NULL, "\t,\r; ");
      if (pch == NULL)
        break;
      header_titles.push_back(pch);
      //update stats if needed
    }
  }

  if (M == 0 && N == 0)
    logstream(LOG_FATAL)<<"Failed to detect matrix size. Please prepare a file named: " << base_filename << ":info with matrix market header, as explained here: http://bickson.blogspot.co.il/2012/12/collaborative-filtering-3rd-generation_14.html " << std::endl;

  logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " << M << " x " << N << ", non-zeros: " << nz << std::endl;

  uint I, J;
  std::vector<float> valarray; valarray.resize(std::max(1, fc.total_features));
  float val;

  if (!fc.hash_strings){
    for (int i=0; i< fc.total_features; i++){
      fc.stats_array[i].minval = 1e100;
      fc.stats_array[i].maxval = -1e100;
    }
  }
    if (limit_rating > 0)
      nz = limit_rating;
    for (size_t i=0; i<nz; i++)
    {

      if (!read_line(f, base_filename, i,I, J, val, valarray, TRAINING))
        logstream(LOG_FATAL)<<"Failed to read line: " <<i<< " in file: " << base_filename << std::endl;

      //avoid self edges
      if (square && I == J)
        continue;

      if (I>= M || J >= N){
        if (i == 0)
          logstream(LOG_FATAL)<<"Failed to parsed first line, there are too many tokens. Did you forget the --has_header_titles=1 flag when file has string column headers?" << std::endl;
        else 
          logstream(LOG_FATAL)<<"Bug: can not add edge from " << I << " to  J " << J << " since max is: " << M <<"x" <<N<<std::endl;
      }
      //calc stats
      L++;
      globalMean += val;
      sharderobj.preprocessing_add_edge(I, square?J:M+J, als_edge_type(val, &valarray[0]));
    }

    sharderobj.end_preprocessing();

    //calc stats
    assert(L > 0);
    for (int i=0; i< fc.total_features; i++){
      fc.stats_array[i].meanval /= L;
    }
    assert(globalMean != 0);
    globalMean /= L;
    logstream(LOG_INFO)<<"Coputed global mean is: " << globalMean << std::endl;

    //print features
    for (int i=0; i< fc.total_features; i++){
      logstream(LOG_INFO) << "Feature " << i << " min val: " << fc.stats_array[i].minval << " max val: " << fc.stats_array[i].maxval << "  mean val: " << fc.stats_array[i].meanval << std::endl;
    }


    FILE * outf = fopen((base_filename + ".gm").c_str(), "w");
    fprintf(outf, "%d\n%d\n%ld\n%d\n%12.8lg", M, N, L, fc.total_features, globalMean);
    for (int i=0; i < fc.total_features; i++){
      fprintf(outf, "%12.8g\n%12.8g\n%12.8g\n", fc.stats_array[i].minval, fc.stats_array[i].maxval, fc.stats_array[i].meanval);
    }
    fclose(outf);

  fclose(f);

  if (fc.hash_strings){
    for (int i=0; i< fc.total_features+2; i++){
      if (fc.node_id_maps[i].string2nodeid.size() == 0)
        logstream(LOG_FATAL)<<"Failed to save feature number : " << i << " no values find in data " << std::endl;
    }
  }
    
  logstream(LOG_INFO) << "Now creating shards." << std::endl;
  // Shard with a specified number of shards, or determine automatically if not defined
  nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));

  return nshards;
}

/* read node features from file */
void read_node_features(std::string base_filename, bool square, feature_control & fc, bool user, bool binary) {
  FILE *f;

  if ((f = fopen(base_filename.c_str(), "r")) == NULL) {
    logstream(LOG_FATAL) << "Could not open file: " << base_filename << ", error: " << strerror(errno) << std::endl;
  }
  binary = true; //TODO
  double_map fmap;
  fc.node_id_maps.push_back(fmap);
  fc.node_features++;
  stats stat;
  fc.stats_array.push_back(stat);

  uint I, J = -1;
  char * linebuf = NULL;
  char linebuf_debug[1024];
  size_t linesize;
  size_t lines = 0;
  size_t tokens = 0;
  float val = 1;

  while(true){
    /* READ LINE */
    int rc = getline(&linebuf, &linesize, f);
    if (rc == -1)
      break;
    strncpy(linebuf_debug, linebuf, 1024);
    lines++;

    /** READ [FROM] */
    char *pch = strtok(linebuf,"\t,\r; ");
    if (pch == NULL)
      logstream(LOG_FATAL)<<"Error reading line " << lines << " [ " << linebuf_debug << " ] " << std::endl;
    I = (uint)get_node_id(pch, user?0:1, lines, true);
    if (I == (uint)-1) //user id was not found in map, so we do not need this users features
      continue;

    if (user)
      assert(I >= 0 && I < M);
    else assert(I>=0  && I< N);


    /** READ USER FEATURES */
    while (pch != NULL){
      pch = strtok(NULL, "\t,\r; ");
      if (pch == NULL)
        break;
      if (binary){
        if (atoi(pch) <= 2)
          continue;
        J = (uint)get_node_id(pch, 2+fc.total_features+fc.node_features-1, lines);
      }
      else { 
        pch = strtok(NULL, "\t\r,;: ");
        if (pch == NULL)
          logstream(LOG_FATAL)<<"Failed to read feture value" << std::endl;
        val = atof(pch);
      }
      set_new(latent_factors_inmem[user? I : I+M].features, J, val);
      tokens++;
      //update stats if needed
    }
  }

  assert(tokens > 0);
  logstream(LOG_DEBUG)<<"Read a total of " << lines << " node features. Tokens: " << tokens << " avg tokens: " << (lines/tokens) 
    << " user? " << user <<  " new entries: " << fc.node_id_maps[2+fc.total_features+fc.node_features-1].string2nodeid.size() << std::endl;
}


/* read node features from file */
void read_node_links(std::string base_filename, bool square, feature_control & fc, bool user, bool binary) {
  FILE *f;

  if ((f = fopen(base_filename.c_str(), "r")) == NULL) {
    logstream(LOG_FATAL) << "Could not open file: " << base_filename << ", error: " << strerror(errno) << std::endl;
  }
  //double_map fmap;
  //fc.node_id_maps.push_back(fmap);
  fc.node_links++;
  //stats stat;
  //fc.stats_array.push_back(stat);

  uint I, J = -1;
  char * linebuf = NULL;
  char linebuf_debug[1024];
  size_t linesize;
  size_t lines = 0;
  size_t tokens = 0;
  float val = 1;

  while(true){
    /* READ LINE */
    int rc = getline(&linebuf, &linesize, f);
    if (rc == -1)
      break;
    strncpy(linebuf_debug, linebuf, 1024);
    lines++;

    /** READ [FROM] */
    char *pch = strtok(linebuf,"\t,\r; ");
    if (pch == NULL)
      logstream(LOG_FATAL)<<"Error reading line " << lines << " [ " << linebuf_debug << " ] " << std::endl;
    I = (uint)get_node_id(pch, user? 0 : 1, lines, true);
    if (I == (uint)-1)//user id was not found in map, we do not need this user link features
      continue; 

    if (user)
      assert(I < (uint)fc.offsets[1]);
    else assert(I < (uint)fc.offsets[2]);

    /** READ TO */  
    pch = strtok(NULL, "\t,\r; ");
      if (pch == NULL)
        logstream(LOG_FATAL)<<"Failed to read to field [ " << linebuf_debug << " ] " << std::endl;

      J = (uint)get_node_id(pch, user? 0 : 1, lines);
      set_new(latent_factors_inmem[user? I : I+M].links, J, val);
      tokens++;
      //update stats if needed
  }

  logstream(LOG_DEBUG)<<"Read a total of " << lines << " node features. Tokens: " << tokens << " user? " << user <<  " new entries: " << fc.node_id_maps[user? 0 : 1].string2nodeid.size() << std::endl;
}


#include "rmse.hpp"




/**
  compute validation rmse
  */
  void validation_rmse_N(
      float (*prediction_func)(const vertex_data ** array, int arraysize, float rating, double & prediction, vec * psum)
      ,graphchi_context & gcontext, 
      feature_control & fc, 
      bool square = false) {

    assert(fc.total_features <= fc.feature_num);
    if (validation == "")
      return;
      
     if (!file_exists(validation)) {
         logstream(LOG_WARNING) << "Validation file was specified, but not found:" << validation << std::endl;
         return;
    }

    FILE *f = NULL;
    size_t nz;   

    detect_matrix_size(validation, f, Me, Ne, nz);
    if (f == NULL){
       logstream(LOG_WARNING)<<"Failed to open validation data. Skipping."<<std::endl;
       return;
    }

    if ((M > 0 && N > 0) && (Me != M || Ne != N))
      logstream(LOG_WARNING)<<"Input size of validation matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

    Le = nz;

    last_validation_rmse = dvalidation_rmse;
    dvalidation_rmse = 0;   
    double validation_error = 0;
    
    std::vector<float> valarray; valarray.resize(fc.total_features);
    uint I, J;
    float val;

    for (size_t i=0; i<nz; i++)
    {
      if (!read_line(f, test, i, I, J, val, valarray, VALIDATION))
        logstream(LOG_FATAL)<<"Failed to read line: " << i << " in file: " << validation << std::endl;

      double prediction;
      vertex_data ** node_array = new vertex_data*[calc_feature_node_array_size(I,J)];
      for (int k=0; k< calc_feature_node_array_size(I,J); k++)
        node_array[k] = NULL;
      vec sum;
      compute_prediction(I, J, val, prediction, &valarray[0], prediction_func, &sum, node_array);
      delete [] node_array;
      dvalidation_rmse += pow(prediction - val, 2);
      if (calc_error) 
        if ((prediction < cutoff && val > cutoff) || (prediction > cutoff && val < cutoff))
          validation_error++;
    }

    fclose(f);

    assert(Le > 0);
    dvalidation_rmse = sqrt(dvalidation_rmse / (double)Le);
    std::cout<<"  Validation RMSE: " << std::setw(10) << dvalidation_rmse;
    if (!calc_error)
      std::cout << std::endl;
    else std::cout << " Validation error: " << std::setw(10) << validation_error/Le << std::endl;
    if (halt_on_rmse_increase && dvalidation_rmse > last_validation_rmse && gcontext.iteration > 0){
      logstream(LOG_WARNING)<<"Stopping engine because of validation RMSE increase" << std::endl;
      gcontext.set_last_iteration(gcontext.iteration);
    }
  }



/* compute predictions for test data */
void test_predictions_N(
    float (*prediction_func)(const vertex_data ** node_array, int node_array_size, float rating, double & predictioni, vec * sum), 
    feature_control & fc, 
    bool square = false) {
  FILE * f = NULL;
  uint Me, Ne;
  size_t nz;   

  if (test == ""){
    logstream(LOG_INFO)<<"No test file was found, skipping test predictions " << std::endl;
    return;
  }
    
    if (!file_exists(test)) {
        logstream(LOG_ERROR) << " test predictions file was specified but not found: " << test << std::endl;
        return;
    }

  detect_matrix_size(test, f, Me, Ne, nz);
  if (f == NULL){
     logstream(LOG_WARNING)<<"Failed to open test file. Skipping " << std::endl;
     return;
  }
  if ((M > 0 && N > 0 ) && (Me != M || Ne != N))
    logstream(LOG_FATAL)<<"Input size of test matrix must be identical to training matrix, namely " << M << "x" << N << std::endl;

  FILE * fout = open_file((test + ".predict").c_str(),"w");

  MM_typecode matcode;
  mm_set_array(&matcode);
  mm_write_banner(fout, matcode);
  mm_write_mtx_array_size(fout ,nz, 1); 
  
  std::vector<float> valarray; valarray.resize(fc.total_features);
  float val;
  double prediction;
  uint I,J;

  for (uint i=0; i<nz; i++)
  {

    if (!read_line(f, test, i, I, J, val, valarray, TEST))
      logstream(LOG_FATAL)<<"Failed to read line: " <<i << " in file: " << test << std::endl;

    vertex_data ** node_array = new vertex_data*[calc_feature_node_array_size(I,J)];
    vec sum;
    compute_prediction(I, J, val, prediction, &valarray[0], prediction_func, &sum, node_array);
    fprintf(fout, "%d %d %12.8lg\n", I+1, J+1, prediction);
    delete[] node_array;
  }
  fclose(f);
  fclose(fout);

  logstream(LOG_INFO)<<"Finished writing " << nz << " predictions to file: " << test << ".predict" << std::endl;
}





float gensgd_predict(const vertex_data** node_array, int node_array_size,
    const float rating, double& prediction, vec* sum){

  vec sum_sqr = zeros(D);
  *sum = zeros(D);
  prediction = globalMean;
  assert(!std::isnan(prediction));
  for (int i=0; i< node_array_size; i++)
    prediction += node_array[i]->bias;
  assert(!std::isnan(prediction));

  for (int j=0; j< D; j++){
    for (int i=0; i< node_array_size; i++){
      sum->operator[](j) += node_array[i]->pvec[j];
      assert(sum->operator[](j) < 1e5);
      sum_sqr[j] += pow(node_array[i]->pvec[j],2);
    }
    prediction += 0.5 * (pow(sum->operator[](j),2) - sum_sqr[j]);
    assert(!std::isnan(prediction));
  }
  //truncate prediction to allowed values
  prediction = std::min((double)prediction, maxval);
  prediction = std::max((double)prediction, minval);
  //return the squared error
  float err = rating - prediction;
  assert(!std::isnan(err));
  return err*err; 

}
float gensgd_predict(const vertex_data** node_array, int node_array_size,
    const float rating, double & prediction){
  vec sum;
  return gensgd_predict(node_array, node_array_size, rating, prediction, &sum);
}


void init_gensgd(bool load_factors_from_file){

  srand(time(NULL));
  int nodes = M+N+num_feature_bins()+fc.last_item*M;
  latent_factors_inmem.resize(nodes);
  int howmany = calc_feature_num();
  logstream(LOG_DEBUG)<<"Going to calculate: " << howmany << " offsets." << std::endl;
  fc.offsets.resize(howmany);
  get_offsets(fc.offsets);
  assert(D > 0);
  if (!load_factors_from_file){
  double factor = 0.1/sqrt(D);
#pragma omp parallel for
  for (int i=0; i< nodes; i++){
    latent_factors_inmem[i].pvec = (debug ? 0.1*ones(D) : (::randu(D)*factor));
  }
  }
}


void training_rmse_N(int iteration, graphchi_context &gcontext, bool items = false){
  last_training_rmse = dtraining_rmse;
  dtraining_rmse = 0;
  size_t total_errors = 0;
  int start = 0;
  int end = M;
  if (items){
    start = M;
    end = M+N;
  }
  dtraining_rmse = sum(rmse_vec);
  if (calc_error)
    total_errors = sum(errors_vec);
  dtraining_rmse = sqrt(dtraining_rmse / pengine->num_edges());
  if (calc_error)
  std::cout<< std::setw(10) << mytimer.current_time() << ") Iteration: " << std::setw(3) <<iteration<<" Training RMSE: " << std::setw(10)<< dtraining_rmse << " Train err: " << std::setw(10) << (total_errors/(double)L);
  else 
  std::cout<< std::setw(10) << mytimer.current_time() << ") Iteration: " << std::setw(3) <<iteration<<" Training RMSE: " << std::setw(10)<< dtraining_rmse;
}

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct GensgdVerticesInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  /*
   *  Vertex update function - computes the least square step
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
    if (fc.last_item && gcontext.iteration == 0){
      if (is_user(vertex.id()) && vertex.num_outedges() > 0) { //user node. find the last rated item and store it. we assume items are sorted by time!
        vertex_data& user = latent_factors_inmem[vertex.id()]; 
        int max_time = 0;
        for(int e=0; e < vertex.num_outedges(); e++) {
          const edge_data & edge = vertex.outedge(e)->get_data();
          if (edge.features[0] >= max_time){ //first feature is time
            max_time = (int)ceil(edge.features[0]);
            user.last_item = vertex.outedge(e)->vertex_id() - M;
          }
        }
      }
      else if (is_user(vertex.id()) && vertex.num_outedges() == 0)
        vertex_with_no_edges++;
      return;
    } 

    //go over all user nodes
    if (is_user(vertex.id())){
      vertex_data& user = latent_factors_inmem[vertex.id()]; 
      assert(user.last_item >= 0 && user.last_item < (int)N);


      //go over all observed ratings
      for(int e=0; e < vertex.num_outedges(); e++) {
        int howmany = calc_feature_node_array_size(vertex.id(), vertex.outedge(e)->vertex_id()-M);
        vertex_data ** node_array = new vertex_data*[howmany];
        for (int i=0; i< howmany; i++)
          node_array[i] = NULL;

        const edge_data & data = vertex.outedge(e)->get_data();
        float rui = data.weight;
        double pui;
        vec sum;

        //compute current prediction
        rmse_vec[omp_get_thread_num()] += compute_prediction(vertex.id(), vertex.outedge(e)->vertex_id()-M, rui ,pui, (float*)data.features, gensgd_predict, &sum, node_array);
        if (calc_error)
          if ((pui < cutoff && rui > cutoff) || (pui > cutoff && rui < cutoff))
            errors_vec[omp_get_thread_num()]++;
        float eui = pui - rui;

        //update global mean bias
        globalMean -= gensgd_rate1 * (eui + gensgd_reg0 * globalMean);

        //update node biases and  vectors
        for (int i=0; i < calc_feature_node_array_size(vertex.id(), vertex.outedge(e)->vertex_id()-M); i++){

          double gensgd_rate;    
          if (i == 0)  //user
            gensgd_rate = gensgd_rate1;
          else if (i == 1) //item
            gensgd_rate = gensgd_rate2;
          else if (i < 2+fc.total_features) //rating features
            gensgd_rate = gensgd_rate3;
          else if (i < 2+fc.total_features+fc.node_features) //user and item features
            gensgd_rate = gensgd_rate4;
          else 
            gensgd_rate = gensgd_rate5; //last item
          
          node_array[i]->bias -= gensgd_rate * (eui + gensgd_regw* node_array[i]->bias);
          assert(!std::isnan(node_array[i]->bias));
          assert(node_array[i]->bias < 1e3);
 
          vec grad =  sum - node_array[i]->pvec;
          node_array[i]->pvec -= gensgd_rate * (eui*grad + gensgd_regv * node_array[i]->pvec);
          assert(!std::isnan(node_array[i]->pvec[0]));
          assert(node_array[i]->pvec[0] < 1e3);
        }
        delete[] node_array;

      }


    }

  };

  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    if (iteration == 1 && vertex_with_no_edges > 0)
      logstream(LOG_WARNING)<<"There are " << vertex_with_no_edges << " users without ratings" << std::endl;
    gensgd_rate1 *= gensgd_mult_dec;
    gensgd_rate2 *= gensgd_mult_dec;
    gensgd_rate3 *= gensgd_mult_dec;
    gensgd_rate4 *= gensgd_mult_dec;
    gensgd_rate5 *= gensgd_mult_dec;
    training_rmse_N(iteration, gcontext);
    validation_rmse_N(&gensgd_predict, gcontext, fc);
  };

  /**
   * Called before an iteration is started.
   */
  void before_iteration(int iteration, graphchi_context &gcontext) {
    rmse_vec = zeros(gcontext.execthreads);
    if (calc_error)
      errors_vec = zeros(gcontext.execthreads);
  }



};


void output_gensgd_result(std::string filename) {
  MMOutputter_mat<vertex_data> mmoutput(filename + "_U.mm", 0, M+N+num_feature_bins(), "This file contains Gensgd output matrices. In each row D factors of a single user node, then item nodes, then features", latent_factors_inmem);
  MMOutputter_vec<vertex_data> mmoutput_bias(filename + "_U_bias.mm", 0, num_feature_bins(), BIAS_POS, "This file contains Gensgd output bias vector. In each row a single user bias.", latent_factors_inmem);
  MMOutputter_scalar gmean(filename + "_global_mean.mm", "This file contains Gensgd global mean which is required for computing predictions.", globalMean);

  logstream(LOG_INFO) << " GENSGD output files (in matrix market format): " << filename << "_U.mm" << ",  "<< filename <<  "_global_mean.mm, " << filename << "_U_bias.mm "  <<std::endl;
}

int main(int argc, const char ** argv) {

  print_copyright();  

  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("als-tensor-inmemory-factors");

  //specific command line parameters for gensgd
  gensgd_rate1 = get_option_float("gensgd_rate1", gensgd_rate1);
  gensgd_rate2 = get_option_float("gensgd_rate2", gensgd_rate2);
  gensgd_rate3 = get_option_float("gensgd_rate3", gensgd_rate3);
  gensgd_rate4 = get_option_float("gensgd_rate4", gensgd_rate4);
  gensgd_rate5 = get_option_float("gensgd_rate5", gensgd_rate5);
  gensgd_regw = get_option_float("gensgd_regw", gensgd_regw);
  gensgd_regv = get_option_float("gensgd_regv", gensgd_regv);
  gensgd_reg0 = get_option_float("gensgd_reg0", gensgd_reg0);
  gensgd_mult_dec = get_option_float("gensgd_mult_dec", gensgd_mult_dec);
  fc.last_item = get_option_int("last_item", fc.last_item);
  fc.hash_strings = get_option_int("rehash", fc.hash_strings);
  user_file = get_option_string("user_file", user_file);
  user_links = get_option_string("user_links", user_links);
  item_file = get_option_string("item_file", item_file);
  file_columns = get_option_int("file_columns"); //get the number of columns in the edge file
  if (file_columns < 3)
    logstream(LOG_FATAL)<<"You must have at least 3 columns in input file: [from] [to] [value] on each line"<<std::endl;
  D = get_option_int("D", D);
  fc.from_pos = get_option_int("from_pos", fc.from_pos);
  fc.to_pos = get_option_int("to_pos", fc.to_pos);
  fc.val_pos = get_option_int("val_pos", fc.val_pos);
  limit_rating = get_option_int("limit_rating", limit_rating);
  calc_error = get_option_int("calc_error", calc_error);
  has_header_titles = get_option_int("has_header_titles", has_header_titles);
  fc.rehash_value = get_option_int("rehash_value", fc.rehash_value);
  cutoff = get_option_float("cutoff", cutoff);

  parse_command_line_args();
  parse_implicit_command_line();

  std::string string_features = get_option_string("features", fc.default_feature_str);
  if (string_features != ""){
    char * pfeatures = strdup(string_features.c_str());
    char * pch = strtok(pfeatures, ",\n\r\t ");
    int node = atoi(pch);
    if (node < 0 || node >= MAX_FEATURES+3)
      logstream(LOG_FATAL)<<"Feature id using the --features=XX command should be non negative, starting from zero"<<std::endl;
    fc.feature_selection[node] = true;
    fc.total_features++;
    while ((pch = strtok(NULL, ",\n\r\t "))!= NULL){
      node = atoi(pch);
      if (node < 0 || node >= MAX_FEATURES+3)
        logstream(LOG_FATAL)<<"Feature id using the --features=XX command should be non negative, starting from zero"<<std::endl;
      fc.feature_selection[node] = true;
      fc.total_features++;
    }
  }
  fc.node_id_maps.resize(2+fc.total_features);
  fc.stats_array.resize(fc.total_features);

  int nshards = convert_matrixmarket_N<edge_data>(training, false, fc, limit_rating);
 
  init_gensgd(load_factors_from_file);
  if (user_file != "")
    read_node_features(user_file, false, fc, true, false);
  if (item_file != "")
    read_node_features(item_file, false, fc, false, false);
  if (user_links != "")
    read_node_links(user_links, false, fc, true, false);

  if (has_header_titles && header_titles.size() == 0)
    logstream(LOG_FATAL)<<"Please delete temp files (using : \"rm -f " << training << ".*\") and run again" << std::endl;

  logstream(LOG_INFO) <<"Total selected features: " << fc.total_features << " : " << std::endl;
  for (int i=0; i < MAX_FEATURES+3; i++)
    if (fc.feature_selection[i])
      logstream(LOG_INFO)<<"Selected feature: " << std::setw(3) << i << " : " << (has_header_titles? header_titles[i] : "") <<std::endl;
  logstream(LOG_INFO)<<"Target variable " << std::setw(3) << fc.val_pos << " : " << (has_header_titles? header_titles[fc.val_pos] : "") <<std::endl;
  logstream(LOG_INFO)<<"From            " << std::setw(3) << fc.from_pos<< " : " << (has_header_titles? header_titles[fc.from_pos] : "") <<std::endl;
  logstream(LOG_INFO)<<"To              " << std::setw(3) << fc.to_pos  << " : " << (has_header_titles? header_titles[fc.to_pos] : "") <<std::endl;

  if (fc.node_features){
    int last_offset = fc.node_id_maps.size();
    int toadd = 0;
    for (int i = last_offset - fc.node_features; i < last_offset; i++){
      toadd += fc.node_id_maps[i].string2nodeid.size();
    }
    logstream(LOG_DEBUG)<<"Going to add " << toadd << std::endl;
    vertex_data data;
    for (int i=0; i < toadd; i++){
      data.pvec = zeros(D);
      for (int j=0; j < D; j++)
        data.pvec[j] = drand48();
      latent_factors_inmem.push_back(data);
    }
    fc.offsets.resize(calc_feature_num());
    get_offsets(fc.offsets);
  }
  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, D);
    vec user_bias =      load_matrix_market_vector(training +"_U_bias.mm", false, true);
    assert(user_bias.size() == num_feature_bins());
    for (uint i=0; num_feature_bins(); i++){
      latent_factors_inmem[i].bias = user_bias[i];
    }
    vec gm = load_matrix_market_vector(training + "_global_mean.mm", false, true);
    globalMean = gm[0];
  }


  /* Run */
  GensgdVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output test predictions in matrix-market format */
  output_gensgd_result(training);
  test_predictions_N(&gensgd_predict, fc);    

  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}
