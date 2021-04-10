#ifndef HELPER_H
#define HELPER_H

#include <cstdlib>
#include "accpar.h"

void parser(char * filename,
            const int batch_size,
            Layer *& layer_list,
            int & num_layer);

void print_partitions(int num_h,
                      int num_layer,
                      Layer * layer_list_original,
                      int ** partitions,
                      double * com_cost);

void gen_resnet (const int num_layer,
                 Layer *& layer_list,
                 const int batch_size);

#endif