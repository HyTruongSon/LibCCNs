#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <ctype.h>

using namespace std;

// Global variables
const string data_name = "WebKB";
const vector<string> sub_data_name = {"cornell", "texas", "washington", "wisconsin"};
const vector<string> suffix = {"cites", "content"};

// Merging
void merge(const string suffix) {
	const string output_fn = data_name + "/" + data_name + "." + suffix;
	ofstream output(output_fn.c_str(), ios::out);
	for (int i = 0; i < sub_data_name.size(); ++i) {
		const string input_fn = data_name + "/" + sub_data_name[i] + "." + suffix;
		ifstream input(input_fn.c_str(), ios::in);
		string str;
		while (getline(input, str)) {
			output << str << endl;
		}
		cout << endl;
		input.close(); 
	}
	output.close();
}

// Main program
int main(int argc, char **argv) {
	srand(time(0));
	for (int i = 0; i < suffix.size(); ++i) {
		merge(suffix[i]);
	}
	return 0;
}