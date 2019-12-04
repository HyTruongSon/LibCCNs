% Program: ENZYMES Data Processing
% Author: Hy Truong Son
% Position: PhD student
% Institution: Department of Computer Science, The University of Chicago
% Email: hytruongson@uchicago.edu, sonpascal93@gmail.com
% Website: http://people.inf.elte.hu/hytruongson/
% Copyright 2016 (c) Hy Truong Son. All rights reserved. Only for academic
% purposes.

% Data structure
% --------------
% First line: Number of samples (enzymes)
% For each sample (enzyme):
% - Number of vertices N
% - The next line contains N numbers that is label of the vertex
% - The next N lines, each line contains a number of adjacent vertices,
% then a list of (vertex, weight). By default, weight = 1.
% - The last line is the type of the enzyme

function [] = enzymes_data_processing()
    load('ENZYMES.mat', 'ENZYMES', 'lenzymes');
    nSamples = size(ENZYMES, 2);
    fid = fopen('ENZYMES.dat', 'w');
    fprintf(fid, '%d\n', nSamples);
    for sample = 1 : nSamples
        N = size(ENZYMES(sample).al, 1);
        fprintf(fid, '%d\n', N);
        for i = 1 : N
            label = ENZYMES(sample).nl.values(i);
            fprintf(fid, '%d ', label);
        end
        fprintf(fid, '\n');
        for i = 1 : N
            nAdj = size(ENZYMES(sample).al{i, 1}, 2);
            fprintf(fid, '%d ', nAdj);
            for j = 1 : nAdj
                vertex = ENZYMES(sample).al{i, 1}(j);
                weight = 1;
                fprintf(fid, '%d %d ', vertex, weight);
            end
            fprintf(fid, '\n');
        end
        type = lenzymes(sample);
        fprintf(fid, '%d\n', type);
    end
    fclose(fid);
end
