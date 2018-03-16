CC=clang++
CFLAGS=-std=c++11 --stdlib=libc++
LIBS=-lpng

AR=ar
ARCHIVE=libtinypng.a

SRC=$(filter-out unit_test.cc, $(wildcard *.cc))
OBJ=$(SRC:.cc=.o)
HDR=$(wildcard *.h)

SYSTEM=$(shell uname -s)

ifeq ($(SYSTEM), Linux)
# LIBS += -Wl,-lstdc++
endif

.PHONY: all debug unit clean

all: CFLAGS += -O3
all: $(OBJ)
	$(AR) sr $(ARCHIVE) $^

debug: CFLAGS += -DDEBUG -g
debug: $(OBJ)
	$(AR) sr $(ARCHIVE) $^

unit: CFLAGS += -DDEBUG -g
unit: $(OBJ) unit_test.o
	$(CC) $(CFLAGS) $(LIBS) $^ -o $@

%.o: %.cc $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o unit $(ARCHIVE)
