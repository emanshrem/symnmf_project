CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
OBJS = symnmf.o
TARGET = symnmf

all: $(TARGET) ext

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c -o symnmf.o

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: ext
ext:
	python3 setup.py build_ext --inplace
