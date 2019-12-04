% Program: MUTAG Data Processing
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

function [] = mutag_data_processing()
    load('MUTAG.mat', 'MUTAG', 'lmutag');
    nSamples = size(MUTAG, 2);
    
    fid = fopen('MUTAG.dat', 'w');
    fprintf(fid, '%d\n', nSamples);
    
    for sample = 1 : nSamples
        N = size(MUTAG(sample).al, 1);
        fprintf(fid, '%d\n', N);
        for i = 1 : N
            label = MUTAG(sample).nl.values(i);
            fprintf(fid, '%d ', label);
        end
        fprintf(fid, '\n');
        
        adj = zeros(N, N);
        M = size(MUTAG(sample).el.values, 1);
        for i = 1 : M
            u = MUTAG(sample).el.values(i, 1);
            v = MUTAG(sample).el.values(i, 2);
            w = MUTAG(sample).el.values(i, 3);
            adj(u, v) = w;
            adj(v, u) = w;
        end
        
        for i = 1 : N
            nAdj = size(MUTAG(sample).al{i, 1}, 2);
            fprintf(fid, '%d ', nAdj);
            for j = 1 : nAdj
                vertex = MUTAG(sample).al{i, 1}(j);
                weight = adj(i, vertex);
                fprintf(fid, '%d %d ', vertex, weight);
            end
            fprintf(fid, '\n');
        end
        type = lmutag(sample);
        fprintf(fid, '%d\n', type);
    end
    fclose(fid);
end
