#include <iostream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <assert.h>

using namespace std;

static int Index(const int x1, const int x2, const int X2) {
	return x1 * X2 + x2;
}

static int Index(const int x1, const int x2, const int x3, const int X2, const int X3) {
	return (x1 * X2 + x2) * X3 + x3;
}

static int Index(const int x1, const int x2, const int x3, const int x4, const int X2, const int X3, const int X4) {
	return ((x1 * X2 + x2) * X3 + x3) * X4 + x4;
}