% Program: NCI1 Data Processing
% Author: Hy Truong Son
% Position: PhD student
% Institution: Department of Computer Science, The University of Chicago
% Email: hytruongson@uchicago.edu, sonpascal93@gmail.com
% Website: http://people.inf.elte.hu/hytruongson/
% Copyright 2016 (c) Hy Truong Son. All rights reserved. Only for academic
% purposes.

% Data structure
% --------------
% First line: Number of samples
% For each sample:
% - Number of vertices N
% - The next line contains N numbers that is label of the vertex
% - The next N lines, each line contains a number of adjacent vertices,
% then a list of (vertex, weight). By default, weight = 1.
% - The last line is the type of the enzyme

function [] = nci1_data_processing()
    load('NCI1.mat', 'NCI1', 'lnci1');
    nSamples = size(NCI1, 2);
    
    fid = fopen('NCI1.dat', 'w');
    fprintf(fid, '%d\n', nSamples);
    
    for sample = 1 : nSamples
        N = size(NCI1(sample).al, 1);
        fprintf(fid, '%d\n', N);
        for i = 1 : N
            label = NCI1(sample).nl.values(i);
            fprintf(fid, '%d ', label);
        end
        fprintf(fid, '\n');
        
        for i = 1 : N
            nAdj = size(NCI1(sample).al{i, 1}, 2);
            fprintf(fid, '%d ', nAdj);
            for j = 1 : nAdj
                vertex = NCI1(sample).al{i, 1}(j);
                weight = NCI1(sample).el.values{i}(j);
                fprintf(fid, '%d %d ', vertex, weight);
            end
            fprintf(fid, '\n');
        end
        type = lnci1(sample);
        fprintf(fid, '%d\n', type);
    end
    fclose(fid);
end
