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
string data_name;
int train;
int val;
int test;
string cites_fn;
string content_fn;
string log_fn;
string vertex_fn;
string edge_fn;
string feature_fn;
string class_fn;
string meta_fn;
string train_fn;
string val_fn;
string test_fn;
vector<string> paper_name;
vector< pair<string, string> > cites;
vector< pair<int, int> > edges;

// Log
ofstream LOG;

// Flags
vector< pair<string, string> > flags;

// Parsing flags
void parse_flags(int argc, char **argv) {
	flags.clear();
	for (int i = 1; i < argc; ++i) {
		int start = 1;
		if (argv[i][1] == '-') {
			start = 2;
		}
		string name = "";
		for (int j = start; j < strlen(argv[i]); ++j) {
			if (argv[i][j] != '=') {
				name += argv[i][j];
			} else {
				break;
			}
		}
		string value = "";
		for (int j = start + name.length() + 1; j < strlen(argv[i]); ++j) {
			value += argv[i][j];
		}
		flags.push_back(make_pair(name, value));
	}
	for (int i = 0; i < flags.size(); ++i) {
		for (int j = i + 1; j < flags.size(); ++j) {
			if (flags[i].first == flags[j].first) {
				cerr << "Flag error: argument " << (i + 1) << " and argument " << (j + 1) << " overlap." << endl;
			}
		}
	}
}

// Get the flag's value
string get_flag(const string &name) {
	for (int i = 0; i < flags.size(); ++i) {
		if (name == flags[i].first) {
			return flags[i].second;
		}
	}
	cerr << "Flag error: could not find flag " << name << endl;
	return "";
}

// Parse string
vector<string> parse_string(const string &str) {
	vector<string> result;
	result.clear();
	int i = 0;
	while (i < str.length()) {
		if (isspace(str[i])) {
			++i;
			continue;
		}
		string word = "";
		for (int j = i; j < str.length(); ++j) {
			if (!isspace(str[j])) {
				word += str[j];
			} else {
				break;
			}
		}
		result.push_back(word);
		i += word.length();
	}
	return result;
}

// Read the citation
void read_cites(const string &file_name) {
	cout << "Read the citation from file " << file_name << endl;
	LOG << "Read the citation from file " << file_name << endl;

	cites.clear();
	ifstream file(file_name.c_str(), ios::in);
	string str1, str2;
	while (file >> str1) {
		file >> str2;
		cites.push_back(make_pair(str1, str2));
	}
	file.close();

	vector<string> name;
	name.clear();
	for (int i = 0; i < cites.size(); ++i) {
		name.push_back(cites[i].first);
		name.push_back(cites[i].second);
	}
	sort(name.begin(), name.end());
	paper_name.clear();
	int i = 0;
	while (i < name.size()) {
		paper_name.push_back(name[i]);
		int next = i;
		for (int j = i + 1; j < name.size(); ++j) {
			if (name[j] == name[i]) {
				next = j;
			} else {
				break;
			}
		}
		i = next + 1;
	}

	const double density = (double)(cites.size()) / (double)(paper_name.size());

	cout << "Number of papers: " << paper_name.size() << endl;
	cout << "Number of citations: " << cites.size() << endl;
	cout << "Density: " << density << endl;

	LOG << "Number of papers: " << paper_name.size() << endl;
	LOG << "Number of citations: " << cites.size() << endl;
	LOG << "Density: " << density << endl;
}

// Write .vertex file
void write_vertex(const string &file_name) {
	cout << "Write to file " << file_name << endl;
	LOG << "Write to file " << file_name << endl;

	ofstream file(file_name.c_str(), ios::out);
	file << "Number of papers:" << endl;
	file << paper_name.size() << endl;
	for (int i = 0; i < paper_name.size(); ++i) {
		file << "Index:" << endl << i << endl;
		file << "Paper name:" << endl << paper_name[i] << endl;
	}
	file.close();
}

// Binary search
int search(const string &str, const vector<string> &paper_name) {
	int l = 0;
	int r = paper_name.size() - 1;
	while (l <= r) {
		const int mid = (l + r) / 2;
		if (str == paper_name[mid]) {
			return mid;
		}
		if (str < paper_name[mid]) {
			r = mid - 1;
		} else {
			l = mid + 1;
		}
	}
	return -1;
}

// Write .edge file
void write_edge(const string &file_name) {
	edges.clear();
	for (int i = 0; i < cites.size(); ++i) {
		const int u = search(cites[i].first, paper_name);
		const int v = search(cites[i].second, paper_name);
		assert(u != -1);
		assert(v != -1);
		edges.push_back(make_pair(u, v));
	}
	assert(edges.size() == cites.size());

	cout << "Write to file " << file_name << endl;
	LOG << "Write to file " << file_name << endl;

	ofstream file(file_name.c_str(), ios::out);
	file << "Number of citations:" << endl;
	file << edges.size() << endl;
	for (int i = 0; i < edges.size(); ++i) {
		file << "Index:" << endl << i << endl;
		file << "From:" << endl << edges[i].first << endl;
		file << "To:" << endl << edges[i].second << endl;
	}
	file.close();
}

// Read content
vector<int> read_content(const string &content_fn, const string &feature_fn, const string &class_fn, const string &meta_fn) {
	cout << "Read content from file " << content_fn << endl;
	cout << "Write feature to file " << feature_fn << endl;

	LOG << "Read content from file " << content_fn << endl;
	LOG << "Write feature to file " << feature_fn << endl;	

	int nVocab;
	int count_line = 0;

	vector<int> indices;
	vector<string> classes;
	indices.clear();
	classes.clear();

	ifstream content(content_fn.c_str(), ios::in);
	ofstream feature(feature_fn.c_str(), ios::out);
	string line;
	while (getline(content, line)) {
		vector<string> words = parse_string(line);
		const int index = search(words[0], paper_name);
		assert(index != -1);
		++count_line;
		if (count_line > 1) {
			assert(words.size() == nVocab + 2);
		} else {
			nVocab = words.size() - 2;
		}
		indices.push_back(index);
		classes.push_back(words[words.size() - 1]);
		int nActive = 0;
		for (int i = 1; i <= words.size() - 2; ++i) {
			if (atof(words[i].c_str()) > 0.0) {
				++nActive;
			}
		}
		feature << "Index:" << endl << index << endl;
		feature << "Number of words active:" << endl << nActive << endl;
		feature << "Words active:" << endl;
		for (int i = 1; i <= words.size() - 2; ++i) {
			if (atof(words[i].c_str()) > 0.0) {
				feature << (i - 1) << " ";
			}
		}
		feature << endl;
		feature << "Words' values:" << endl;
		for (int i = 1; i <= words.size() - 2; ++i) {
			float value = atof(words[i].c_str());
			if (value > 0.0) {
				feature << value << " ";
			}
		}
		feature << endl;
	}
	content.close();
	feature.close();

	assert(count_line == indices.size());
	assert(count_line == classes.size());
	// assert(count_line == paper_name.size());

	vector<string> unique_classes;
	unique_classes.clear();
	for (int i = 0; i < classes.size(); ++i) {
		bool found = false;
		for (int j = 0; j < unique_classes.size(); ++j) {
			if (classes[i] == unique_classes[j]) {
				found = true;
				break;
			}
		}
		if (!found) {
			unique_classes.push_back(classes[i]);
		}
	}
	sort(unique_classes.begin(), unique_classes.end());

	cout << "Write feature to file " << class_fn << endl;
	LOG << "Write feature to file " << class_fn << endl;

	ofstream class_file(class_fn.c_str(), ios::out);
	for (int i = 0; i < indices.size(); ++i) {
		const int c = search(classes[i], unique_classes);
		assert(c != -1);
		class_file << "Index:" << endl << indices[i] << endl;
		class_file << "Class:" << endl << c << endl;
	}
	class_file.close();

	cout << "Write meta data to file " << meta_fn << endl;
	LOG << "Write meta data to file " << meta_fn << endl;

	ofstream meta(meta_fn.c_str(), ios::out);
	meta << "Number of vertices (papers with features):" << endl << indices.size() << endl;
	meta << "Number of papers:" << endl << paper_name.size() << endl;
	meta << "Number of papers without any features (excluded):" << endl << (paper_name.size() - indices.size()) << endl;
	meta << "Number of edges:" << endl << edges.size() << endl;
	meta << "Vocabulary size:" << endl << nVocab << endl;
	meta << "Number of classes:" << endl << unique_classes.size() << endl;
	meta << "Classes:" << endl;
	for (int i = 0; i < unique_classes.size(); ++i) {
		meta << unique_classes[i] << endl;
	}
	meta << "Density:" << endl << ((double)(edges.size()) / (double)(paper_name.size())) << endl;
	meta.close();

	return indices;
}

// Save set
void save_set(const vector<int> &set, const int &percent, const string &file_name) {
	ofstream file(file_name.c_str(), ios::out);
	file << "Number of examples (vertices):" << endl << set.size() << endl;
	file << "Percentage:" << endl << percent << endl;
	file << "Indices of the examples (vertices):" << endl;
	for (int i = 0; i < set.size(); ++i) {
		file << set[i] << endl;
	}
	file.close();
}

// Write the train, val, test
void write_train_val_test(const vector<int> &indices, const int &train, const int &val, const int &test, const string &train_fn, const string &val_fn, const string &test_fn) {
	int nData = indices.size();
	int nTrain = train * nData / 100;
	int nVal = val * nData / 100;
	int nTest = nData - nTrain - nVal;

	int *reorder = new int [nData];
	for (int i = 0; i < nData; ++i) {
		reorder[i] = indices[i];
	}
	for (int i = 0; i < nData; ++i) {
		int j = rand() % nData;
		swap(reorder[i], reorder[j]);
	}

	vector<int> train_set;
	vector<int> val_set;
	vector<int> test_set;
	
	train_set.clear();
	val_set.clear();
	test_set.clear();

	int count = 0;
	for (int i = 0; i < nTrain; ++i) {
		train_set.push_back(reorder[count]);
		++count;
	}
	for (int i = 0; i < nVal; ++i) {
		val_set.push_back(reorder[count]);
		++count;
	}
	for (int i = 0; i < nTest; ++i) {
		test_set.push_back(reorder[count]);
		++count;
	}
	assert(count == nData);

	sort(train_set.begin(), train_set.end());
	sort(val_set.begin(), val_set.end());
	sort(test_set.begin(), test_set.end());

	cout << "Save training set to file " << train_fn << endl;
	LOG << "Save training set to file " << train_fn << endl;
	save_set(train_set, train, train_fn);

	cout << "Save validation set to file " << val_fn << endl;
	LOG << "Save validation set to file " << val_fn << endl;
	save_set(val_set, val, val_fn);

	cout << "Save testing set to file " << test_fn << endl;
	LOG << "Save testing set to file " << test_fn << endl;
	save_set(test_set, test, test_fn);

	delete[] reorder;
}

// Main program
int main(int argc, char **argv) {
	srand(time(0));

	parse_flags(argc, argv);
	data_name = get_flag("data_name");
	train = atoi(get_flag("train").c_str());
	val = atoi(get_flag("val").c_str());
	test = atoi(get_flag("test").c_str());

	log_fn = data_name + "/" + data_name + ".log";
	cites_fn = data_name + "/" + data_name + ".cites";
	content_fn = data_name + "/" + data_name + ".content";
	vertex_fn = data_name + "/" + data_name + ".vertex";
	edge_fn = data_name + "/" + data_name + ".edge";
	feature_fn = data_name + "/" + data_name + ".feature";
	class_fn = data_name + "/" + data_name + ".class";
	meta_fn = data_name + "/" + data_name + ".meta";
	train_fn = data_name + "/" + data_name + ".train";
	val_fn = data_name + "/" + data_name + ".val";
	test_fn = data_name + "/" + data_name + ".test";

	LOG.open(log_fn.c_str(), ios::out);

	cout << "Data name: " << data_name << endl;
	cout << "Log: " << log_fn << endl;
	cout << "Cites: " << cites_fn << endl;
	cout << "Content: " << content_fn << endl;
	cout << "Vertex: " << vertex_fn << endl;
	cout << "Edge: " << edge_fn << endl;
	cout << "Feature: " << feature_fn << endl;
	cout << "Class: " << class_fn << endl;
	cout << "Meta: " << meta_fn << endl;
	cout << "Train: " << train_fn << endl;
	cout << "Val: " << val_fn << endl;
	cout << "Test: " << test_fn << endl;

	LOG << "Data name: " << data_name << endl;
	LOG << "Log: " << log_fn << endl;
	LOG << "Cites: " << cites_fn << endl;
	LOG << "Content: " << content_fn << endl;
	LOG << "Vertex: " << vertex_fn << endl;
	LOG << "Edge: " << edge_fn << endl;
	LOG << "Feature: " << feature_fn << endl;
	LOG << "Class: " << class_fn << endl;
	LOG << "Meta: " << meta_fn << endl;
	LOG << "Train: " << train_fn << endl;
	LOG << "Val: " << val_fn << endl;
	LOG << "Test: " << test_fn << endl;

	read_cites(cites_fn);
	write_vertex(vertex_fn);
	write_edge(edge_fn);
	vector<int> indices = read_content(content_fn, feature_fn, class_fn, meta_fn);
	write_train_val_test(indices, train, val, test, train_fn, val_fn, test_fn);

	LOG.close();
	return 0;
}