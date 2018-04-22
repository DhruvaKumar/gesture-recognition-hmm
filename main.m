clc;
clear;
addpath(genpath('/home/dhruva/Documents/Learning in Robotics/3-HMM'));
%% TO TEST, PLEASE RUN ONLY THE LAST SECTION
%% AND NOT ENTIRE SCRIPT! OTHERWISE IT'LL START TRAINING TOO!

% ==================== Description ==========================
% 
% Author: Dhruva Kumar
% 
% Algorithm:
% - Quantize the observation via kmeans
% - Learn the model parameters {pi, A, B} by the Baum Welch procedure
% - Evaluation on test data
% 
% Initial parameters for the model (chosen via cross validation):
%  - N = 10 | M = 20 
%  - HMM structure: left to right with the last state looped to the first
%  - A, B randomly initialized | pi = [1,0,0....] starts from the first
%  state
%
% Convergence criteria:
% The difference of log likelihood of P(O|lambda) < 0.1
%
% Vectorization:
% - Unvectorized Baum welch for 5 sequences & for 1 iteration
% 4.514 seconds
% - Vectorized Baum welch for 5 sequences & for 1 iteration
% 0.86 seconds
% ==============================================================

%% Quantize the observations: kmeans clustering

%quantData
load models/quantized_observations

%% Learn the model parameters: EM Baum Welch

model = struct('Name', {}, 'pi', {}, 'A', {}, 'B', {}, 'centroids', {}, ...
        'costFunction', {});

fprintf ('Learning model... \n');
for c = 1:length(quantizedObs)
    model(c).Name = quantizedObs(c).Name;
    model(c).centroids = quantizedObs(c).centroids;
    fprintf('Gesture %d: %s \n',c, model(c).Name);
    
    % init model {pi_prev, A_prev, B_prev}
    N = 10; M = 20;
    % fully connected model
%     pi_prev = 1/N * ones(N,1);
%     A_prev = rand(N,N); 
%     A_prev = A_prev ./ repmat(sum(A_prev), N,1); % normalize
%     B_prev = rand(M,N); 
%     B_prev = B_prev ./ repmat(sum(B_prev), M,1); % normalize

    % left-right model
    pi_prev = zeros(N,1);
    pi_prev(1) = 1;
    A_prev = diag(rand(N,1)) + diag(rand(N-1,1),-1); % band diagonal
    A_prev(1,N) = rand(1,1); % loop left-right
    A_prev = A_prev ./ repmat(sum(A_prev), N,1); % normalize
    B_prev = rand(M,N); 
    B_prev = B_prev ./ repmat(sum(B_prev), M,1); % normalize
    
    
    % load inital_model
    log_p_prev = 1;
    costFunc = zeros(50,1);
    
    O_multiple = quantizedObs(c).X_gesture_quant; % {Lx1}
    
    % EM: Baum Welch
    iter = 200;
    for i = 1:iter
        % E step: forward backward
        [alpha, beta, c_alpha_multiple, log_p] = ...
            hmm_fb_multiple(pi_prev, A_prev, B_prev, O_multiple);
        % M step: update step
        [pi, A, B] = hmm_update_multiple (alpha, beta, c_alpha_multiple,...
                                            O_multiple, A_prev, B_prev);
    
        % convergence: max (log_p_O_model) 
        costFunc(i) = log_p;
        changeLog2 = abs(log_p - log_p_prev) / (1+abs(log_p_prev));
        changeLog = abs(log_p - log_p_prev);
        fprintf('Iteration %d: %f | changeLog: %f | changeLog2: %f \n', ...
            i, log_p, changeLog, changeLog2);

        if (changeLog < 0.1)
            % use the previous values
            pi = pi_prev;
            A = A_prev;
            B = B_prev;
            break;
        end
        log_p_prev = log_p;
        pi_prev = pi;
        A_prev = A;
        B_prev = B;
    end % em iteration
    
    % update model
    model(c).pi = pi;
    model(c).A = A;
    model(c).B = B;
    model(c).costFunction = costFunc;
    
    fprintf('---------------------------------------------\n');
end % gesture /class

save ('models/modelNEW.mat', 'model');
    
%% Evaluation on train set

% clc;
% load models/model
% 
% dirstruct = dir('train/');
% 
% for i = 1:length(dirstruct)
%     if (dirstruct(i).isdir && ~strcmp(dirstruct(i).name, '.') && ~strcmp(dirstruct(i).name, '..'))
%         subdirstruct = dir(strcat('train/',dirstruct(i).name,'/*.txt'));
%         for j = 1:length(subdirstruct)
%            fpath =  strcat('train/',dirstruct(i).name,'/',subdirstruct(j).name);
%            
%            X_test = cleanData(fpath);
%            
%            % for each gesture
%            log_p = zeros(length(model),1);
%            for c = 1:length(model)
%                 % discretize test observation
%                 X_test_quant = quantObs(X_test(:,2:end), model(c).centroids);
%                 [~, ~, ~, log_p(c)] = hmm_fb (model(c).pi, model(c).A,...
%                                                    model(c).B, X_test_quant);
%            end
% 
%            [~, ind] = max(log_p);
%             temp = log_p;
%             temp(ind)=nan;
%             [~, second] = max(temp);
%             temp(second) = nan;
%             [~, third] = max(temp);
%             fprintf('Actual: %s | predicted: %s | %s | %s\n',dirstruct(i).name,...
%         model(ind).Name, model(second).Name, model(third).Name);
%         end   
%      fprintf('---------------------------------------------\n');
%     end
% end


%% Evaluation on sample test set

% % clc; clear;
% load models/model11
% 
% dirstruct = dir('sample test/*.txt');
% actual = 3;
% for i = 1:length(dirstruct)
% %     i=1;
%     fpath = strcat('sample test/',dirstruct(i).name);
%     % read and clean test observation
%     X_test = cleanData(fpath);
%     
%     % for each gesture
%     log_p = zeros(length(model),1);
%     for c = 1:length(model)
%         % discretize test observation
%         X_test_quant = quantObs(X_test(:,2:end), model(c).centroids);
%         [~, ~, ~, log_p(c)] = hmm_fb (model(c).pi, model(c).A,...
%                                            model(c).B, X_test_quant);
%     end
%     
%     [~, ind] = max(log_p);
%     temp = log_p;
%     temp(ind)=nan;
%     [~, second] = max(temp);
%     temp(second) = nan;
%     [~, third] = max(temp);
%     fprintf('Actual: %s | predicted: %s | %s | %s\n',model(actual).Name,...
%         model(ind).Name, model(second).Name, model(third).Name);
%     % normalize negative values
% %     temp =  log_p + abs(min(log_p));
% %     acc = temp / norm(temp);
% %     hf = figure(i); 
% %     hold on, bar(acc), bar(acc.*(acc==max(acc)), 'g'), hold off;
%     %saveas(hf,strcat('results/multiple_',num2str(i),'.png'));
% end

%% Evaluation on test set
clc;
load models/model

fprintf('Actual gesture | predicted: most likely guess | second | third\n');
fprintf('---------------------------------------------\n');

dirstruct = dir('test/single/*.txt');

fprintf('For single... \n');
% confusion = zeros(6,6);
for i = 1:length(dirstruct)
    fpath = strcat('test/single/',dirstruct(i).name);
    % read and clean test observation
    X_test = cleanData(fpath);
    
    % for each gesture
    log_p = zeros(length(model),1);
    for c = 1:length(model)
        % discretize test observation
        X_test_quant = quantObs(X_test(:,2:end), model(c).centroids);
        [~, ~, ~, log_p(c)] = hmm_fb (model(c).pi, model(c).A,...
                                           model(c).B, X_test_quant);
    end
    
    [~, ind] = max(log_p);
    temp = log_p;
    temp(ind)=nan;
    [~, second] = max(temp);
    temp(second) = nan;
    [~, third] = max(temp);
    fprintf('%s: | predicted: %s | %s | %s\n',dirstruct(i).name, ...
        model(ind).Name, model(second).Name, model(third).Name);
end
fprintf('---------------------------------------------\n');

fprintf('For multiple... \n');
dirstruct = dir('test/multiple/*.txt');
% actual = [];
for i = 1:length(dirstruct)
    fpath = strcat('test/multiple/',dirstruct(i).name);
    % read and clean test observation
    X_test = cleanData(fpath);
    
    % for each gesture
    log_p = zeros(length(model),1);
    for c = 1:length(model)
        % discretize test observation
        X_test_quant = quantObs(X_test(:,2:end), model(c).centroids);
        [~, ~, ~, log_p(c)] = hmm_fb (model(c).pi, model(c).A,...
                                           model(c).B, X_test_quant);
    end
    
    [~, ind] = max(log_p);
    temp = log_p;
    temp(ind)=nan;
    [~, second] = max(temp);
    temp(second) = nan;
    [~, third] = max(temp);
    fprintf('%s: | predicted: %s | %s | %s\n',dirstruct(i).name,...
        model(ind).Name, model(second).Name, model(third).Name);
end






    
    
    
    
    