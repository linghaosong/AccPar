#include "accpar.h"
#include <iostream>

using std::cout;


Matrix::Matrix(double r, 
               double c) {
    row_num = r;
    col_num = c;
}


void Matrix::Devide_By_Row() {
    row_num /= 2.0;
}


void Matrix::Devide_By_Col() {
    col_num /= 2.0;
}


double Matrix::Matrix_Size() {
    return row_num * col_num;
}


double Matrix::Com_Cost_Matrix(TENSOR_PARTITION from_p,
                               TENSOR_PARTITION to_p) {
    if (from_p == ROW) {
        if (to_p == ROW) {
            return 0.0;
        }else if (to_p == COLUMN) {
            return Matrix_Size() / 4.0;
        }else if (to_p == DUPLICATE) {
            return Matrix_Size() / 2.0;
        }else if (to_p == REDUCE) {
            cout << "Illegal communication patterns\n";
            exit(1);
        }
    }else if (from_p == COLUMN) {
        if (to_p == ROW) {
            return Matrix_Size() / 4.0;
        }else if (to_p == COLUMN) {
            return 0.0;
        }else if (to_p == DUPLICATE) {
            return Matrix_Size() / 2.0;
        }else if (to_p == REDUCE) {
            cout << "Illegal communication patterns\n";
            exit(1);
        }
    }else if (from_p == DUPLICATE) {
        if (to_p == ROW) {
            return 0.0;
        }else if (to_p == COLUMN) {
            return 0.0;
        }else if (to_p == DUPLICATE) {
            return 0.0;
        }else if (to_p == REDUCE) {
            cout << "Illegal communication patterns\n";
            exit(1);
        }
    }else if (from_p == REDUCE) {
        cout << "Illegal communication patterns\n";
        exit(1);
        //if (to_p == ROW) 
        //    return Matrix_Size() / 2.0;
        //}else if (to_p == COLUMN) {
        //    return Matrix_Size() / 2.0;
        //}else if (to_p == DUPLICATE) {
        //    return Matrix_Size();
        //}else if (to_p == REDUCE) {
        //    return 0.0;
        //}
    }

    cout << "Illegal communication patterns\n";
    exit(1);
}


Matrix & Matrix::operator= (const Matrix & rhs) {
    this->row_num = rhs.row_num;
    this->col_num = rhs.col_num;
    return *this;
}


Layer::Layer(string n_str,
             FCCONV l_t,
             double batch_size,
             double num_channel_in,
             double featuremap_height_in,
             double featuremap_width_in,
             double num_channel_out,
             double filter_height,
             double filter_width,
             double featuremap_height_out,
             double featuremap_width_out,
             int ps) {
    name = n_str;
    layer_type = l_t;
    pool_size = ps;

    if (l_t == FC) {
        input_matrix = Matrix(batch_size, num_channel_in);
        weight_matrix = Matrix(num_channel_in, num_channel_out);
        output_matrix = Matrix(batch_size, num_channel_out);
            
        input_matrix_derivation = input_matrix;
        weight_matrix_derivation = weight_matrix;
        output_matrix_derivation = output_matrix;
    } else {
        input_matrix = Matrix(batch_size, num_channel_in);
        weight_matrix = Matrix(num_channel_in, num_channel_out);
        output_matrix = Matrix(batch_size, num_channel_out);
            
        input_matrix_derivation = input_matrix;
        weight_matrix_derivation = weight_matrix;
        output_matrix_derivation = output_matrix;
    }

    kernel_size = filter_height * filter_width;
    ifmap_size = featuremap_height_in * featuremap_width_in;
    ofmap_size = featuremap_height_out * featuremap_width_out;
}


Layer & Layer::operator= (const Layer & rhs) {
    this->name = rhs.name;
    this->layer_type = rhs.layer_type;
        
    this->input_matrix = rhs.input_matrix;
    this->weight_matrix = rhs.weight_matrix;
    this->output_matrix = rhs.output_matrix;
        
    this->input_matrix_derivation = rhs.input_matrix_derivation;
    this->weight_matrix_derivation = rhs.weight_matrix_derivation;
    this->output_matrix_derivation = rhs.output_matrix_derivation;
        
    this->pool_size = rhs.pool_size;
    this->kernel_size = rhs.kernel_size;
    this->ifmap_size = rhs.ifmap_size;
    this->ofmap_size = rhs.ofmap_size;
        
    return *this;
}


double Layer::Com_Cost_Inter_Layer(BASIC_TYPE prev_layer,
                                   BASIC_TYPE this_layer) {
    if (prev_layer == TYPE_I) {
        if (this_layer == TYPE_I) {
            return 0.0;
        }else if (this_layer == TYPE_II) {
            return 0.5 * 0.5 * (input_matrix.Matrix_Size() + input_matrix_derivation.Matrix_Size()) * ifmap_size;
        }else if (this_layer == TYPE_III) {
            return 0.5 * input_matrix.Matrix_Size() * ifmap_size;
        }else{
            cout << "Illegal inter layer communication patterns\n";
            exit(1);
        }
    }else if (prev_layer == TYPE_II) {
        if (this_layer == TYPE_I) {
            return 0.5 * input_matrix_derivation.Matrix_Size() * ifmap_size;
        }else if (this_layer == TYPE_II) {
            return 0.5 * input_matrix_derivation.Matrix_Size() * ifmap_size;
        }else if (this_layer == TYPE_III) {
            return 0.0;
        }else{
            cout << "Illegal inter layer communication patterns\n";
            exit(1);
        }
    }else if (prev_layer == TYPE_III) {
        if (this_layer == TYPE_I) {
            return 0.5 * 0.5 * (input_matrix.Matrix_Size() + input_matrix_derivation.Matrix_Size()) * ifmap_size;
        }else if (this_layer == TYPE_II) {
            return 0.0;
        }else if (this_layer == TYPE_III) {
            return 0.5 * input_matrix.Matrix_Size() * ifmap_size;
        }else{
            cout << "Illegal inter layer communication patterns\n";
            exit(1);
        }
    }

    cout << "Illegal inter layer communication patterns\n";
    exit(1);
}


double Layer::Com_Cost_Inter_Layer_Block(BASIC_TYPE from_layer,
                                         BASIC_TYPE out_layer) {
    if (from_layer == TYPE_I) {
        if (out_layer == TYPE_I) {
            return 0.0;
        }else if (out_layer == TYPE_II) {
            return 0.5 * input_matrix.Matrix_Size() * ifmap_size;
        }else if (out_layer == TYPE_III) {
            return 0.5 * 0.5 * (input_matrix.Matrix_Size() + input_matrix_derivation.Matrix_Size()) * ifmap_size;
        }else{
            cout << "Illegal inter layer communication patterns\n";
            exit(1);
        }
    }else if (from_layer == TYPE_II) {
        if (out_layer == TYPE_I) {
            return 0.5 * 0.5 * (input_matrix.Matrix_Size() + input_matrix_derivation.Matrix_Size()) * ifmap_size;
        }else if (out_layer == TYPE_II) {
            return 0.5 * input_matrix.Matrix_Size() * ifmap_size;
        }else if (out_layer == TYPE_III) {
            return 0.0;
        }else{
            cout << "Illegal inter layer communication patterns\n";
            exit(1);
        }
    }else if (from_layer == TYPE_III) {
        if (out_layer == TYPE_I) {
            return 0.5 * input_matrix_derivation.Matrix_Size() * ifmap_size;
        }else if (out_layer == TYPE_II) {
            return 0.0;
        }else if (out_layer == TYPE_III) {
            return 0.5 * input_matrix_derivation.Matrix_Size() * ifmap_size;
        }else{
            cout << "Illegal inter layer communication patterns\n";
            exit(1);
        }
    }
    
    cout << "Illegal inter layer communication patterns\n";
    exit(1);
}


void Layer::partition_by(BASIC_TYPE t) {
    switch (t) {
        case TYPE_I:
            input_matrix.Devide_By_Row();
            input_matrix_derivation.Devide_By_Row();
            output_matrix.Devide_By_Row();
            output_matrix_derivation.Devide_By_Row();
            break;
                
        case TYPE_II:
            input_matrix.Devide_By_Col();
            input_matrix_derivation.Devide_By_Col();
            weight_matrix.Devide_By_Row();
            weight_matrix_derivation.Devide_By_Row();
            break;
                
        case TYPE_III:
            weight_matrix.Devide_By_Col();
            weight_matrix_derivation.Devide_By_Col();
            output_matrix.Devide_By_Col();
            output_matrix_derivation.Devide_By_Col();
            break;
                
        default:
            cout << "partition_by() unknow type\n";
            exit(1);
            break;
    }
}


bool is_layer_short_cut_in(int num_layer, int l) {
    if ((num_layer != 18) && (num_layer != 34) && (num_layer != 50)) {
        cout << "is_layer_short_cut: not resnet\n";
        exit(1);
    }
    
    if (num_layer == 50) {
        return (3 < l) && (l % 3 == 1);
    }else {
        return (2 < l) && (l % 2 == 1);
    }
}


void dp_on_one_hierarchy(int num_layer,
                         Layer * layer_list,
                         int * partitions,
                         double & com_cost_this_hierarchy,
                         const int NUM_TYPES){
    
    double ** cost_table = new double*[num_layer];
    int ** track_table = new int*[num_layer];
    for (int i = 0; i < num_layer; ++i) {
        cost_table[i] = new double[NUM_TYPES];
        track_table[i] = new int[NUM_TYPES];
    }
    
    //initiate track_table
    for (int i = 0; i < NUM_TYPES; ++i) {
        track_table[0][i] = i;    }
    
    //initiate cost_table, intra layer communication
    for (int l = 0; l < num_layer; ++l) {
        for (int i = 0; i < NUM_TYPES; ++i) {
            if (i == 0) {
                cost_table[l][i] = layer_list[l].weight_matrix_derivation.Matrix_Size()
                                    * layer_list[l].kernel_size;
            } else if (i == 1) {
                cost_table[l][i] = layer_list[l].output_matrix.Matrix_Size()
                                    * layer_list[l].ofmap_size;
            } else if (i == 2) {
                cost_table[l][i] = layer_list[l].input_matrix_derivation.Matrix_Size()
                                    * layer_list[l].ifmap_size;
            } else {
                cout << "Unknow type \n";
                exit(1);
            }
        }
    }
    
    bool isAccPar = NUM_TYPES == 3;
    bool isResNet = (num_layer == 18) || (num_layer == 34) || (num_layer == 50);
    
    ///////////////////////////OTHER //////////////////////
    
    //dp
    for (int l = 1; l < num_layer; ++l) { //layer
        for (int t_this = 0; t_this < NUM_TYPES; ++t_this) {//type for layer l
            double min_cost_lt = cost_table[l-1][0] +
            layer_list[l].Com_Cost_Inter_Layer(TYPE_I, BASIC_TYPE_LIST[t_this]);
            int min_pre_type = 0;
            
            for (int t_pre = 0; t_pre < NUM_TYPES; ++t_pre) { //type for layer l-1
                double tmp_cost = cost_table[l-1][t_pre] +
                layer_list[l].Com_Cost_Inter_Layer(BASIC_TYPE_LIST[t_pre], BASIC_TYPE_LIST[t_this]);
                if (isAccPar && isResNet && is_layer_short_cut_in(num_layer, l)) {
                    //trace back 2 layer
                    int t_pre_pre = track_table[l-1][t_pre];
                    
                    if (num_layer != 50) {
                        tmp_cost += layer_list[l].Com_Cost_Inter_Layer_Block(BASIC_TYPE_LIST[t_pre_pre], BASIC_TYPE_LIST[t_this]);
                    } else {
                        //trace back 2 layer
                        int t_pre_pre_pre = track_table[l-2][t_pre_pre];
                        tmp_cost += layer_list[l].Com_Cost_Inter_Layer_Block(BASIC_TYPE_LIST[t_pre_pre_pre], BASIC_TYPE_LIST[t_this]);
                    }
                    
                }
                if (min_cost_lt > tmp_cost) {
                    min_cost_lt = tmp_cost;
                    min_pre_type = t_pre;
                }
            }
            
            //update trace_table
            cost_table[l][t_this] += min_cost_lt;
            track_table[l][t_this] = min_pre_type;
        }
    }
    
    double min_cost = cost_table[num_layer-1][0];
    int min_last_type = 0;
    
    for (int t = 1; t < NUM_TYPES; ++t) {
        if (min_cost > cost_table[num_layer-1][t]) {
            min_cost = cost_table[num_layer-1][t];
            min_last_type = t;
        }
    }
    
    com_cost_this_hierarchy = min_cost;
    partitions[num_layer-1] = min_last_type;
    for (int l = num_layer-2; l >= 0; --l) {
        partitions[l] = track_table[l+1][min_last_type];
        min_last_type = track_table[l+1][min_last_type];
    }
    
    for (int i = 0; i < num_layer; ++i) {
        delete [] cost_table[i];
        delete [] track_table[i];
    }
    delete [] track_table;
    delete [] cost_table;
}

void h_partition(int num_h,
                 int num_layer,
                 Layer * layer_list_original,
                 int ** partitions,
                 double * com_cost,
                 const int NUM_TYPES) {
    Layer * layer_list = new Layer[num_layer];
    for (int i = 0; i < num_layer; ++i) {
        layer_list[i] = layer_list_original[i];
    }
    
    for (int h = 0; h < num_h; ++h) {
        dp_on_one_hierarchy(num_layer,
                            layer_list,
                            partitions[h],
                            com_cost[h],
                            NUM_TYPES);
        for (int l = 0; l < num_layer; ++l) {
            layer_list[l].partition_by(BASIC_TYPE_LIST[partitions[h][l]]);
        }
    }
    
    delete [] layer_list;
}
