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
const string old_cites_fn = "Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab";
const string new_cites_fn = "Pubmed-Diabetes/Pubmed-Diabetes.cites";

const string old_content_fn = "Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab";
const string new_content_fn = "Pubmed-Diabetes/Pubmed-Diabetes.content";

const string vocab_fn = "Pubmed-Diabetes/Pubmed-Diabetes.vocab";

vector<string> papers;
vector< vector< pair<string, float> > > contents;
vector<string> labels;
vector<string> vocab;

// Convert cites
string standardize(const string str) {
	assert(str[0] == 'p');
	assert(str[1] == 'a');
	assert(str[2] == 'p');
	assert(str[3] == 'e');
	assert(str[4] == 'r');
	assert(str[5] == ':');
	string result = "";
	for (int i = 6; i < str.length(); ++i) {
		result += str[i];
	}
	return result;
}

void convert_cites() {
	ofstream output(new_cites_fn.c_str(), ios::out);
	ifstream input(old_cites_fn.c_str(), ios::in);
	string str;
	getline(input, str);
	getline(input, str);
	while (input >> str) {
		string paper1, bar, paper2;
		input >> paper1 >> bar >> paper2;
		output << standardize(paper1) << " " << standardize(paper2) << endl;
	}
	output.close();
	input.close();
}

// Convert content
vector<string> get_words(const string &str) {
	vector<string> result;
	result.clear();
	int i = 0;
	while (i < str.length()) {
		if (isspace(str[i])) {
			++i;
		}
		string word = "";
		for (int j = i; j < str.length(); ++j) {
			if (!isspace(str[j])) {
				word += str[j];
			} else {
				break;
			}
		}
		i += word.length();
		result.push_back(word);
	}
	return result;
}

pair<string, string> get_parts(const string &str) {
	int pos = -1;
	for (int i = 0; i < str.length(); ++i) {
		if (str[i] == '=') {
			pos = i;
			break;
		}
	}
	assert(pos != -1);
	string part1 = "";
	for (int i = 0; i < pos; ++i) {
		part1 += str[i];
	}
	string part2 = "";
	for (int i = pos + 1; i < str.length(); ++i) {
		part2 += str[i];
	}
	return make_pair(part1, part2);
}

void insert_vocab(const string &str) {
	for (int i = 0; i < vocab.size(); ++i) {
		if (str == vocab[i]) {
			return;
		}
	}
	vocab.push_back(str);
}

int search_vocab(const string &str) {
	for (int i = 0; i < vocab.size(); ++i) {
		if (str == vocab[i]) {
			return i;
		}
	}
	return -1;
}

string type_name(const string &str) {
	if (str == "1") {
		return "Experimental";
	}
	if (str == "2") {
		return "Type-1";
	}
	return "Type-2";
}

void convert_content() {
	ifstream input(old_content_fn.c_str(), ios::in);
	papers.clear();
	labels.clear();
	contents.clear();
	vocab.clear();
	string str;
	getline(input, str);
	getline(input, str);
	while (true) {
		getline(input, str);
		if (str.length() == 0) {
			break;
		}
		vector<string> words = get_words(str);
		const int N = words.size();
		
		papers.push_back(words[0]);
		
		pair<string, string> label = get_parts(words[1]);
		assert(label.first == "label");
		assert(((label.second == "1") || (label.second == "2") || (label.second == "3")));
		labels.push_back(label.second);
		
		pair<string, string> summary = get_parts(words[N - 1]);
		assert(summary.first == "summary");
		
		vector< pair<string, float> > content;
		content.clear();
		for (int i = 2; i <= N - 2; ++i) {
			pair<string, string> info = get_parts(words[i]);
			content.push_back(make_pair(info.first, atof(info.second.c_str())));
			insert_vocab(info.first);
		}
		contents.push_back(content);
	}
	input.close();

	sort(vocab.begin(), vocab.end());

	ofstream file(vocab_fn.c_str(), ios::out);
	file << "Number of words in the vocabulary:" << endl;
	file << vocab.size() << endl;
	file << "Vocabulary:" << endl;
	for (int i = 0; i < vocab.size(); ++i) {
		file << vocab[i] << endl;
	}
	file.close();

	const int V = vocab.size();
	float *arr = new float [V];

	ofstream output(new_content_fn.c_str(), ios::out);
	for (int i = 0; i < papers.size(); ++i) {
		output << papers[i] << " ";
		for (int j = 0; j < V; ++j) {
			arr[j] = 0.0;
		}
		for (int j = 0; j < contents[i].size(); ++j) {
			pair<string, float> info = contents[i][j];
			const int index = search_vocab(info.first);
			assert(index != -1);
			assert(index < V);
			arr[index] = info.second;
		}
		for (int j = 0; j < V; ++j) {
			output << arr[j] << " ";
		}
		output << type_name(labels[i]) << endl; 
	}
	input.close();

	delete[] arr;
}

// Main program
int main(int argc, char **argv) {
	srand(time(0));
	convert_cites();
	convert_content();
	return 0;
}