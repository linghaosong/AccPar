INC = -I ./include
GXX_FLAGS = -w -O3 -g


.PHONY: all lib clean
.DEFAULT_GOAL := all

all: lib main

lib: accpar.o helper.o

main: lib main.o
	g++ $(GXX_FLAGS) accpar.o helper.o main.o -o accpar $(INC)

accpar.o: ./src/accpar.cpp
	g++ $(GXX_FLAGS) -c ./src/accpar.cpp $(INC)

helper.o: ./src/helper.cpp
	g++ $(GXX_FLAGS) -c ./src/helper.cpp $(INC)

main.o: ./src/main.cpp
	g++ $(GXX_FLAGS) -c ./src/main.cpp $(INC)

clean:
	rm *.o
	rm accpar
