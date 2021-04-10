#ifndef ACCPAR_H
#define ACCPAR_H

#include <cstdlib>
#include <string>

using std::string;

const double NUM_BYTE_BFLT = 2.0; //B, 16.0 bit

enum FCCONV {FC=0, CONV=1};

enum TENSOR_PARTITION {
    ROW = 0,
    COLUMN = 1,
    DUPLICATE = 2,
    REDUCE = 3
};

const TENSOR_PARTITION TENSOR_PARTITION_LIST[4] = {ROW, COLUMN, DUPLICATE, REDUCE};

//const TENSOR_PARTITION BASIC_TYPES[3][3] = {
//    {ROW, DUPLICATE, ROW},
//    {DUPLICATE, COLUMN, COLUMN},
//    {COLUMN, ROW, DUPLICATE}
//};

enum BASIC_TYPE {
    TYPE_I = 0,
    TYPE_II = 1,
    TYPE_III = 2
};

const BASIC_TYPE BASIC_TYPE_LIST[3] = {TYPE_I, TYPE_II, TYPE_III};

class Matrix {
public:
    double row_num;
    double col_num;

    Matrix(double r=0.0, double c=0.0);

    void Devide_By_Row();

    void Devide_By_Col();

    double Matrix_Size();
    
    double Com_Cost_Matrix(TENSOR_PARTITION from_p,
                           TENSOR_PARTITION to_p);
    
    Matrix & operator= (const Matrix & rhs);
};

class Layer {
public:
    string name;
    FCCONV layer_type;
    
    Matrix input_matrix;
    Matrix weight_matrix;
    Matrix output_matrix;
    
    Matrix input_matrix_derivation;
    Matrix weight_matrix_derivation;
    Matrix output_matrix_derivation;
    
    int pool_size;
    double kernel_size;
    double ifmap_size;
    double ofmap_size;
    
    Layer(string n_str="Layer",
          FCCONV l_t=FC,
          double batch_size=0.0,
          double num_channel_in=0.0,
          double featuremap_height_in=1.0,
          double featuremap_width_in=1.0,
          double num_channel_out=0.0,
          double filter_height=1.0,
          double filter_width=1.0,
          double featuremap_height_out=1.0,
          double featuremap_width_out=1.0,
          int ps=0);
    
    Layer & operator= (const Layer & rhs);
    
    double Com_Cost_Inter_Layer(BASIC_TYPE prev_layer,
                                BASIC_TYPE this_layer);
    
    double Com_Cost_Inter_Layer_Block(BASIC_TYPE from_layer,
                                      BASIC_TYPE out_layer);

    void partition_by(BASIC_TYPE t);
};

bool is_layer_short_cut_in(int num_layer, 
                           int l);

void dp_on_one_hierarchy(int num_layer,
                         Layer * layer_list,
                         int * partitions,
                         double & com_cost_this_hierarchy,
                         const int NUM_TYPES);

void h_partition(int num_h,
                 int num_layer,
                 Layer * layer_list_original,
                 int ** partitions,
                 double * com_cost,
                 const int NUM_TYPES);

#endif