function [x,labels] = generateDataA1Q1(N)
N = 10000;
figure(1), clf,     %colors = 'bm'; markers = 'o+';
labels = zeros(0,N);

% Randomly assign values to labels
for i = 1:N
    rand_num = rand();
    if rand_num < 0.3
        labels(i) = 1;
    elseif rand_num > 0.3 && rand_num < 0.6
        labels(i) = 2;
    elseif rand_num > 0.6 && rand_num < 1
        labels(i) = 3;
    end
end

%disp(labels);

for l = 1:3
    indl = find(labels==l);
    if l == 1
        N0 = length(indl);
        w0 = [0.5,0.5]; 
        mu0 = [3 0;0 3;3 3];
        Sigma0(:,:,1) = eye(3); 
        Sigma0(:,:,2) = eye(3);
        gmmParameters.priors = w0; % priors should be a row vector
        gmmParameters.meanVectors = mu0;
        gmmParameters.covMatrices = Sigma0;
        [x(:,indl),components] = generateDataFromGMM(N0,gmmParameters);
        plot3(x(1,indl(components==1)),x(2,indl(components==1)),x(3,indl(components==1)),'mo'), hold on, 
        plot3(x(1,indl(components==2)),x(2,indl(components==2)),x(3,indl(components==2)),'go'), hold on, 
        
    elseif l == 2
        m1 = [2;2;2]; 
        C1 = eye(3);
        N1 = length(indl);
        x(:,indl) = mvnrnd(m1,C1,N1)';
        plot3(x(1,indl),x(2,indl),x(3,indl),'b+'), hold on,
        axis equal,

    elseif l == 3
        m1 = [2;2;2]; 
        C1 = eye(3);
        N1 = length(indl);
        x(:,indl) = mvnrnd(m1,C1,N1)';
        plot3(x(1,indl),x(2,indl),x(3,indl),'b+'), hold on,
        axis equal,

    end
end
%%%

function [x,labels] = generateDataFromGMM(N,gmmParameters)
% Generates N vector samples from the specified mixture of Gaussians
% Returns samples and their component labels
% Data dimensionality is determined by the size of mu/Sigma parameters
priors = gmmParameters.priors; % priors should be a row vector
meanVectors = gmmParameters.meanVectors;
covMatrices = gmmParameters.covMatrices;
n = size(gmmParameters.meanVectors,1); % Data dimensionality
C = length(priors); % Number of components
x = zeros(n,N); labels = zeros(1,N); 
% Decide randomly which samples will come from each component
u = rand(1,N); thresholds = [cumsum(priors),1];
for l = 1:C
    indl = find(u <= thresholds(l)); Nl = length(indl);
    labels(1,indl) = l*ones(1,Nl);
    u(1,indl) = 1.1*ones(1,Nl); % these samples should not be used again
    x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
end
