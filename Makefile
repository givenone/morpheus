CC = g++
CFLAGS = -g -Wall
SRCS = binaryGradient.cpp
PROG = binary

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

.PHONY: all clean

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

all: $(PROG)

clean:
	rm -f $(OBJS) $(PROG)