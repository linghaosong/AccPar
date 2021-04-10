#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <limits>

#include "accpar.h"
#include "helper.h"

using std::cout;

int main(int argc, char** argv) {
     if (argc < 3) {
     cout << "Usage: ./accpar [layer description file] [resnet deppth]\n";
     return -1;
     }
    
    //const int NUM_TYPES = 2;
    
    char * filename = argv[1];
    double batch_size = 512;//(double) atoi(argv[2]);
    int num_hierarchy = 7;//atoi(argv[3]);
    
    Layer * layer_list = NULL;
    int num_layer = 0;
    
    int num_layer_resnet = atoi(argv[2]);
    if (num_layer_resnet == -1) {
        parser(filename, batch_size, layer_list, num_layer);
    }else{
        num_layer = num_layer_resnet;
        gen_resnet(num_layer, layer_list, batch_size);
    }
    
    int ** partitions = new int*[num_hierarchy];
    for (int i = 0; i < num_hierarchy; ++i) {
        partitions[i] = new int[num_layer];
    }
    
    ///////////////////main content//////////////////////
    double * com_cost = new double[num_hierarchy];
    
    //dp_on_one_hierarchy(num_layer,
    //                    layer_list,
    //                    partitions[0],
    //                    com_cost);
    
    double com_total = 0.0;
    
    cout << "\n######## DATA PAR ########\n";
    for (int h = 0; h < num_hierarchy; ++h) {
        for (int l = 0; l < num_layer; ++l) {
            partitions[h][l] = 0;
        }
    }
    
    print_partitions(num_hierarchy,
                     num_layer,
                     layer_list,
                     partitions,
                     com_cost); 

    
    cout << "\n######## THE TRICK ########\n";
    for (int h = 0; h < num_hierarchy; ++h) {
        for (int l = 0; l < num_layer; ++l) {
            if (layer_list[l].layer_type == CONV) {
                partitions[h][l] = 0;
            } else {
                partitions[h][l] = 1;
            }
        }
    }
    
    print_partitions(num_hierarchy,
                     num_layer,
                     layer_list,
                     partitions,
                     com_cost);

    
    cout << "\n######## HYPAR ########\n";
    h_partition(num_hierarchy,
                num_layer,
                layer_list,
                partitions,
                com_cost,
                2);
    
    print_partitions(num_hierarchy,
                     num_layer,
                     layer_list,
                     partitions,
                     com_cost);
    
    cout << "\n######## ACCPAR ########\n";
    h_partition(num_hierarchy,
                num_layer,
                layer_list,
                partitions,
                com_cost,
                3);
    
    print_partitions(num_hierarchy,
                     num_layer,
                     layer_list,
                     partitions,
                     com_cost);
    
    
    ///////////////////clear memory/////////////////////
    for (int i = 0; i < num_hierarchy; ++i) {
        delete [] partitions[i];
    }
    delete [] com_cost;
    delete [] partitions;
    delete [] layer_list;
    return EXIT_SUCCESS;
}
