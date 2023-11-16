ifndef cuda
cuda := 1
endif

cc := $(shell which g++)
ifeq (${cuda},1)
cc := $(shell which nvcc)
endif
hflags := -std=c++20 -O3
dflags := -std=c++20 -x cu -O3 -Xcompiler "${hflags}" --extended-lambda
flags := ${hflags}
ifeq (${cuda},1)
flags := ${dflags}
endif
main:
	${cc} ${flags} -Ihybrid main.cc -o a.x
