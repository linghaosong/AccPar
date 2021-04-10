#include "helper.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;
using std::setw;
using std::to_string;


void parser(char * filename,
            const int batch_size,
            Layer *& layer_list,
            int & num_layer) {
    FILE * f;
    if ((f = fopen(filename, "r")) == NULL) {
        cout << "Could not open " << filename << endl;
        exit(1);
    }
    
    char chstr1[256];
    char chstr2[256];
    string str1;
    string str2;
    double d_in;
    
    fscanf(f, "%s %d\n", chstr1, &num_layer);
    str1 = string(chstr1);
    if (str1 != "num_layers") {
        cout << "num_layers not specified in the first line\n";
        exit(1);
    }
    
    layer_list = new Layer[num_layer];
    
    for (int i = 0; i < num_layer; ++i) {
        fscanf(f, "%s %s\n", chstr1, chstr2);
        str1 = string(chstr1);
        str2 = string(chstr2);
        
        if (str2 != "FC" && str2 != "CONV") {
            cout << "layer type (FC or CONV) specified for " << str1 << endl;
            exit(1);
        }
        
        if (str2 == "FC") {
            fscanf(f, "%s %lg\n", chstr1, &d_in);
            double in_ch = d_in;
            
            fscanf(f, "%s %lg\n", chstr1, &d_in);
            double out_ch = d_in;
            
            layer_list[i] = Layer(str1,
                                  FC,
                                  batch_size,
                                  in_ch,
                                  1,
                                  1,
                                  out_ch,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1);
        }else {
            fscanf(f, "%s %lg\n", chstr1, &d_in);
            double in_ch = d_in;
            
            fscanf(f, "%s %lg\n", chstr1, &d_in);
            double in_fmap_h = d_in;
            
            fscanf(f, "%s %lg\n", chstr1, &d_in);
            double in_fmap_w = d_in;
            
            fscanf(f, "%s %lg\n", chstr1, &d_in);
            double kernel = d_in;
            
            fscanf(f, "%s %lg\n", chstr1, &d_in);
            double out_ch = d_in;
            
            fscanf(f, "%s %lg\n", chstr1, &d_in);
            double out_fmap_h = d_in;
            
            fscanf(f, "%s %lg\n", chstr1, &d_in);
            double out_fmap_w = d_in;
            
            fscanf(f, "%s %lg\n", chstr1, &d_in);
            int pool = d_in;
            
            layer_list[i] = Layer(str1,
                                  CONV,
                                  batch_size,
                                  in_ch,
                                  in_fmap_h,
                                  in_fmap_w,
                                  out_ch,
                                  kernel,
                                  kernel,
                                  out_fmap_h,
                                  out_fmap_w,
                                  pool);
        }
    }
    
    fclose(f);
}


void print_partitions(int num_h,
                      int num_layer,
                      Layer * layer_list_original,
                      int ** partitions,
                      double * com_cost) {
    bool isResNet = (num_layer == 18) || (num_layer == 34) || (num_layer == 50);
    Layer * layer_list = new Layer[num_layer];
    for (int i = 0; i < num_layer; ++i) {
        layer_list[i] = layer_list_original[i];
    }
    
    for (int h = 0; h < num_h; ++h) {
        double com_this_h = 0.0;
        for (int l = 0; l < num_layer; ++l) {
            // intra com
            if (partitions[h][l] == 0) {
                com_this_h += layer_list[l].weight_matrix_derivation.Matrix_Size() * layer_list[l].kernel_size;
            } else if (partitions[h][l] == 1) {
                com_this_h += layer_list[l].output_matrix.Matrix_Size() * layer_list[l].ofmap_size;
            } else if (partitions[h][l] == 2) {
                com_this_h += layer_list[l].input_matrix_derivation.Matrix_Size() * layer_list[l].ifmap_size;
            } else {
                cout << "Unknow type \n";
                exit(1);
            }
            
            if (l > 0) { // inter comm
                BASIC_TYPE t_this = BASIC_TYPE_LIST[partitions[h][l]];
                BASIC_TYPE t_pre = BASIC_TYPE_LIST[partitions[h][l-1]];
                com_this_h += layer_list[l].Com_Cost_Inter_Layer(t_this, t_pre);
            }
        }
        
        for (int l = 0; l < num_layer; ++l) {
            layer_list[l].partition_by(BASIC_TYPE_LIST[partitions[h][l]]);
        }
        
        double tmp_com_short = 0.0;
        if(isResNet) {
            if (num_layer == 18) {
                for (int l = 1; l < 17; l = l + 2) {
                    BASIC_TYPE t_this = BASIC_TYPE_LIST[partitions[h][l]];
                    BASIC_TYPE t_pre = BASIC_TYPE_LIST[partitions[h][l+2]];
                    tmp_com_short += layer_list[l].Com_Cost_Inter_Layer_Block(t_this, t_pre);
                }
            }else if (num_layer == 34) {
                for (int l = 1; l < 33; l = l + 2) {
                    BASIC_TYPE t_this = BASIC_TYPE_LIST[partitions[h][l]];
                    BASIC_TYPE t_pre = BASIC_TYPE_LIST[partitions[h][l+2]];
                    tmp_com_short += layer_list[l].Com_Cost_Inter_Layer_Block(t_this, t_pre);
                }
            }else if (num_layer == 50) {
                for (int l = 1; l < 49; l = l + 3) {
                    BASIC_TYPE t_this = BASIC_TYPE_LIST[partitions[h][l]];
                    BASIC_TYPE t_pre = BASIC_TYPE_LIST[partitions[h][l+3]];
                    tmp_com_short += layer_list[l].Com_Cost_Inter_Layer_Block(t_this, t_pre);
                }
            }
            //cout << "tmp_com_short " << tmp_com_short << endl;
        }
        com_cost[h] = com_this_h + tmp_com_short;
    }
    
    
    cout << setw(4) << "h";
    for (int i = 0; i < num_layer; ++i) {
        cout << setw(8) << layer_list[i].name;
    }
    cout << endl;
    
    for (int h = 0; h < num_h; ++h) {
        cout << setw(4) << h;
        for (int i = 0; i < num_layer; ++i) {
            cout << setw(8) << partitions[h][i];
        }
        cout << endl;
    }
    
    delete [] layer_list;
}


void gen_resnet (const int num_layer,
                 Layer *& layer_list,
                 const int batch_size) {
    if (num_layer != 18 &&
        num_layer != 34 &&
        num_layer != 50) {
        layer_list = NULL;
        return;
    }
    
    string str1;
    int in_ch;
    int in_fmap_h;
    int in_fmap_w;
    int out_ch;
    int kernel;
    int out_fmap_h;
    int out_fmap_w;
    int pool;
    
    layer_list = new Layer[num_layer];
    in_ch = 3;
    in_fmap_h = 224;
    in_fmap_w = 224;
    out_ch = 64;
    kernel = 7;
    out_fmap_h = 56;
    out_fmap_w = 56;
    pool = 4;
    
    int l = 0;
    layer_list[l] = Layer("cv1",CONV,batch_size,in_ch,
                          in_fmap_h,in_fmap_w,out_ch,
                          kernel,kernel,out_fmap_h,
                          out_fmap_w,pool);
    l++;
    
    int repeat = 0;
    if (num_layer == 18) {
        // conv2x
        repeat = 2;
        for (int i = 0; i < repeat; ++i) {
            str1 = "cv2" + to_string((l+1));
            
            in_ch = 64;
            in_fmap_h = 56;
            in_fmap_w = 56;
            out_ch = 64;
            kernel = 3;
            out_fmap_h = 56;
            out_fmap_w = 56;
            pool = 1;
            
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv2" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
        }
        
        // conv3x
        repeat = 2;
        for (int i = 0; i < repeat; ++i) {
            str1 = "cv3" + to_string((l+1));
            
            in_ch = 128;
            in_fmap_h = 28;
            in_fmap_w = 28;
            out_ch = 128;
            kernel = 3;
            out_fmap_h = 28;
            out_fmap_w = 28;
            pool = 1;
            
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv3" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
        }
        
        // conv4x
        repeat = 2;
        for (int i = 0; i < repeat; ++i) {
            str1 = "cv4" + to_string((l+1));
            
            in_ch = 256;
            in_fmap_h = 14;
            in_fmap_w = 14;
            out_ch = 256;
            kernel = 3;
            out_fmap_h = 14;
            out_fmap_w = 14;
            pool = 1;
            
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv4" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
        }
        
        // conv5x
        repeat = 2;
        for (int i = 0; i < repeat; ++i) {
            str1 = "cv5" + to_string((l+1));
            
            in_ch = 512;
            in_fmap_h = 7;
            in_fmap_w = 7;
            out_ch = 512;
            kernel = 3;
            out_fmap_h = 7;
            out_fmap_w = 7;
            pool = 1;
            
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv5" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
        }
        
        layer_list[l] = Layer("fc",FC,batch_size,2048,1,1,1000,1,1,1,1,1);
    }else if (num_layer == 50) {
        // conv2x
        repeat = 3;
        for (int i = 0; i < repeat; ++i) {
            str1 = "cv2" + to_string((l+1));
            
            in_fmap_h = 56;
            in_fmap_w = 56;
            out_fmap_h = 56;
            out_fmap_w = 56;
            pool = 1;
            
            layer_list[l] = Layer(str1,CONV,batch_size,64,
                                  in_fmap_h,in_fmap_w,64,
                                  1,1,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv2" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,64,
                                  in_fmap_h,in_fmap_w,64,
                                  3,3,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv2" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,64,
                                  in_fmap_h,in_fmap_w,256,
                                  1,1,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
        }
        
        // conv3x
        repeat = 4;
        for (int i = 0; i < repeat; ++i) {
            str1 = "cv3" + to_string((l+1));
            
            in_fmap_h = 28;
            in_fmap_w = 28;
            out_fmap_h = 28;
            out_fmap_w = 28;
            pool = 1;
            
            layer_list[l] = Layer(str1,CONV,batch_size,128,
                                  in_fmap_h,in_fmap_w,128,
                                  1,1,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv3" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,128,
                                  in_fmap_h,in_fmap_w,128,
                                  3,3,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv3" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,128,
                                  in_fmap_h,in_fmap_w,512,
                                  1,1,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
        }
        
        // conv4x
        repeat = 6;
        for (int i = 0; i < repeat; ++i) {
            in_fmap_h = 14;
            in_fmap_w = 14;
            out_fmap_h = 14;
            out_fmap_w = 14;
            pool = 1;
            
            str1 = "cv4" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,256,
                                  in_fmap_h,in_fmap_w,256,
                                  1,1,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv4" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,256,
                                  in_fmap_h,in_fmap_w,256,
                                  3,3,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv4" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,256,
                                  in_fmap_h,in_fmap_w,1024,
                                  1,1,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
        }
        
        // conv5x
        repeat = 3;
        for (int i = 0; i < repeat; ++i) {
            in_fmap_h = 7;
            in_fmap_w = 7;
            out_fmap_h = 7;
            out_fmap_w = 7;
            pool = 1;
            
            str1 = "cv5" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,512,
                                  in_fmap_h,in_fmap_w,512,
                                  1,1,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv5" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,512,
                                  in_fmap_h,in_fmap_w,512,
                                  3,3,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv5" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,512,
                                  in_fmap_h,in_fmap_w,2048,
                                  1,1,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
        }
        
        layer_list[l] = Layer("fc",FC,batch_size,2048,1,1,1000,1,1,1,1,1);
        
    }else if (num_layer == 34){
        // conv2x
        repeat = 3;
        for (int i = 0; i < repeat; ++i) {
            str1 = "cv2" + to_string((l+1));
            
            in_ch = 64;
            in_fmap_h = 56;
            in_fmap_w = 56;
            out_ch = 64;
            kernel = 3;
            out_fmap_h = 56;
            out_fmap_w = 56;
            pool = 1;
            
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv2" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
        }
        
        // conv3x
        repeat = 4;
        for (int i = 0; i < repeat; ++i) {
            str1 = "cv3" + to_string((l+1));
            
            in_ch = 128;
            in_fmap_h = 28;
            in_fmap_w = 28;
            out_ch = 128;
            kernel = 3;
            out_fmap_h = 28;
            out_fmap_w = 28;
            pool = 1;
            
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv3" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
        }
        
        // conv4x
        repeat = 6;
        for (int i = 0; i < repeat; ++i) {
            str1 = "cv4" + to_string((l+1));
            
            in_ch = 256;
            in_fmap_h = 14;
            in_fmap_w = 14;
            out_ch = 256;
            kernel = 3;
            out_fmap_h = 14;
            out_fmap_w = 14;
            pool = 1;
            
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv4" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
        }
        
        // conv5x
        repeat = 3;
        for (int i = 0; i < repeat; ++i) {
            str1 = "cv5" + to_string((l+1));
            
            in_ch = 512;
            in_fmap_h = 7;
            in_fmap_w = 7;
            out_ch = 512;
            kernel = 3;
            out_fmap_h = 7;
            out_fmap_w = 7;
            pool = 1;
            
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
            
            str1 = "cv5" + to_string((l+1));
            layer_list[l] = Layer(str1,CONV,batch_size,in_ch,
                                  in_fmap_h,in_fmap_w,out_ch,
                                  kernel,kernel,out_fmap_h,
                                  out_fmap_w,pool);
            l++;
        }
        
        layer_list[l] = Layer("fc",FC,batch_size,2048,1,1,1000,1,1,1,1,1);
    }
}