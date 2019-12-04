rm -f process_data
g++ process_data.cpp -o process_data
train=60
val=20
test=20
./process_data --data_name=citeseer --train=$train --val=$val --test=$test
./process_data --data_name=cora --train=$train --val=$val --test=$test
./process_data --data_name=WebKB --train=$train --val=$val --test=$test
./process_data --data_name=Pubmed-Diabetes --train=$train --val=$val --test=$test