

% params=[4, 18, 2, 3, 6, 3, 6, 5, 56, 0.1, 0.1, 0.1]; %情况1
% params=[4, 18, 2, 3, 6, 3, 6, 5, 56, 0.2, 0.2, 0.2]; %情况2
% params=[4, 18, 2, 3, 6, 3, 30, 5, 56, 0.1, 0.1, 0.1]; %情况3
% params=[4, 18, 1, 1, 6, 2, 30, 5, 56, 0.2, 0.2, 0.2]; %情况4
% params=[4, 18, 8, 1, 6, 2, 10, 5, 56, 0.1, 0.2, 0.1]; %情况5
% params=[4, 18, 2, 3, 6, 3, 10, 40, 56, 0.05, 0.05, 0.05]; %情况6
% C1, C2, Cj1, Cj2, Cz, Cjc, Cd, Cc, P, p1, p2, p_z

clc;clear;
paramCases = {
    [4, 18, 2, 3, 6, 3, 6, 5, 56, 0.1, 0.1, 0.1];   % 情况1
    [4, 18, 2, 3, 6, 3, 6, 5, 56, 0.2, 0.2, 0.2];   % 情况2
    [4, 18, 2, 3, 6, 3, 30, 5, 56, 0.1, 0.1, 0.1];  % 情况3
    [4, 18, 1, 1, 6, 2, 30, 5, 56, 0.2, 0.2, 0.2];  % 情况4
    [4, 18, 8, 1, 6, 2, 10, 5, 56, 0.1, 0.2, 0.1];  % 情况5
    [4, 18, 2, 3, 6, 3, 10, 40, 56, 0.05, 0.05, 0.05] % 情况6
    };
w = @(delta) objective(delta, params);

lb = zeros(1, 6);
ub = ones(1, 6);

% 定义线性不等式约束 A*delta <= b
A = [
    0 0 0 0 -1 1;
    0 0 0 1 0 1;
    1 0 0 0 1 0;
    0 1 0 0 0 1];
b = [0; 0; 1; 1];

% 调用遗传算法求解
options = optimoptions('ga', ...
    'PopulationSize', 200, ...             % 增加种群大小
    'MaxGenerations', 400, ...             % 增加最大代数
    'FunctionTolerance', 1e-6, ...         % 减小目标函数容差
    'ConstraintTolerance', 1e-6, ...       % 减小约束容差
    'MaxStallGenerations', 100);        % 增加最大停滞代数
    
IntCon = 1:6;

for caseNum = 1:6
    params = paramCases{caseNum};

    switch caseNum
        case 1
            disp('情况1');
        case 2
            disp('情况2');
        case 3
            disp('情况3');
        case 4
            disp('情况4');
        case 5
            disp('情况5');
        case 6
            disp('情况6');
    end

    % 定义目标函数
    w = @(delta) objective(delta, params);

    % 调用遗传算法进行优化
    [x, fval] = ga(w, 6, [], [], [], [], lb, ub, [], IntCon,options);

    % 输出结果
    fprintf('情况%d的优化结果:\n', caseNum);
    disp('策略结果:');
    disp(x);
    disp('单位利润:');
    disp(-fval);
end

function W = objective(delta, params)
C1 = params(1);
C2 = params(2);
Cj1 = params(3);
Cj2 = params(4);
Cz = params(5);
Cjc = params(6);
Cd = params(7);
Cc = params(8);
P = params(9);
p1 = params(10);
p2 = params(11);
p_z = params(12);

if delta(1)
    delta(5)=0;
end

if delta(2)
    delta(6)=0;
end

if (delta(5)) || (delta(6))
    delta(4)=1;
end



St1 = C1 + C2 + delta(1) * Cj1 + delta(2) * Cj2 - (delta(1) * p1 * C2 + delta(2) * p2 * C1 + ...
    delta(1) * delta(2) * p1 * (1 - p2) * C2 + delta(1) * delta(2) * (1 - p1) * C1);

Pc = (1 - delta(1)) * (1 - delta(2)) * (1 - (1 - p_z) * (1 - p1) * (1 - p2)) + ...
    (1 - delta(1)) * delta(2) * (p1 + p_z * (1 - p1)) + ...
    delta(1) * (1 - delta(2)) * (p2 + p_z * (1 - p2)) + ...
    delta(1) * delta(2) * p_z;

St2 = Cz + delta(3) * Cjc + (1 - delta(3)) * Pc * Cd - (1 - Pc) * P;

St3 = Pc * delta(4) * Cc;

p5 = p1/Pc ;
p6 = p2/Pc ;

Pnc = (1 - delta(5)) * (1 - delta(6)) * (1 - (1 - p_z) * (1 - p5) * (1 - p6)) + ...
    (1 - delta(5)) * delta(6) * (p5 + p_z * (1 - p5)) + ...
    delta(5) * (1 - delta(6)) * (p6 + p_z * (1 - p6)) + ...
    delta(5) * delta(6) * p_z;

P_st4=delta(5)*(1-delta(6))*(1-delta(5))+ ...
    delta(6)*(1-delta(5))*(1-delta(6))+ ...
    delta(5)*delta(6)*(1-p5-p6+p5*p6)+ ...
    (1-delta(5))*(1-delta(6));

St4 = delta(4) * (delta(5) * Cj1 + delta(6) * Cj2 - (delta(5) * p5 * C2 + delta(6) * p6 * C1 + ...
    delta(5) * delta(6) * p5 * (1 - p6) * C2 + delta(5) * delta(6) * (1 - p5) * C1)) - ...
    P_st4 * (Cz + delta(3) * Cjc + (1 - delta(3)) * Pnc * Cd - (1 - Pnc) * P);

W = St1 + (delta(1) * (1 - p1) + delta(2) * (1 - p2) + delta(1) * delta(2) * (1 - p1 - p2 + p1 * p2)+(1-delta(1)*(1-delta(2)))) * (St2 + St3 + St4);
end
