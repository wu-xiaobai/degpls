function W= PLSRGGr(X0,Y0,p,beta)
n = size(X0,2);


Cxx = X0'*X0+beta*eye(n); 

Cxy = X0'*Y0;
C = Cxy *Cxy';
%%
rand('state',0);
W = orth(rand(n,p))*0.001;
%%
    problem.M  = grassmanngeneralizedfactory(n, p,Cxx);
    problem.cost = @cost;
    function f = cost(W)
    f = -0.5*trace(W'*C*W);
    end
    problem.egrad = @egrad;
    function G = egrad(W)
    G = -C*W;
    end
%checkgradient(problem);% pause;
%checkhessian(problem); pause;
 
  [W, costw, info1, options1] = conjugategradient(problem, W);   
    
 % [W, costw, info1, options1] = steepestdescent(problem, W);
    
end