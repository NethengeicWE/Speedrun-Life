# 速通Matlab
* 类似于python，但是在数学计算方面更为成熟，比如信号处理，求导与积分，更容易生成图表
* 有python接口，但似乎无法接受键盘输入，能正常生成图表
## 常见函数
通用计算表示
```m
c=2.3; r=5.1; 
A=3.4; B=1.5;   % 味大，无需多盐
c^3-sqrt(r*A)-5*B
.*              % 逐元素相乘
& | ~ xor       % 与或非,异或
exp();          % e的指数次方
syms x;         % x作为符号变量，可被微积分，各种输入
sub(f,x,2)      % f(x)中将x=2代入
eval(f)         % 符号变量被赋值后对f进行计算
expand(f)       % 展开f
factor(f)       % 因式分解
collect(f)      % 整理幂次
solve(f,x)      % 求解f式中x的值

rand(m,n)       % 生成m行n列个在0-1间的伪随机数
                % (b-a)*r+a：   将rand的随机范围限制在[a,b]上
randn(m,n)      % 正态分布  
mean(x)         % 均值
std(x)          % 标准差
median(x)       % 中值
cumsum(x)       % 逐个累加
```
通用编程格式
```m
% for循环
for index = start : step : end 
    command statements 
end 

% while循环
while condition 
    command statements 
end 

% if条件判断
if condition1 
    command statements 
elseif condition2 
    command statements 
elseif condition3 
    command statements 
end 

% switch判断
switch op 
case ‘+’ 
    x=num1+num2 
case ‘-‘ 
    x=num1-num2 
case ‘*’ 
    x=num1*num2 
otherwise 
    x=num1/num2 
end 

% 函数
function [out1, out2] = filename(input)
    % [filename]函数才能被外界使用,其他命名的函数只能在本文件下使用 这段注释能在help中看到
    command
end
```
微积分
```m
int(f,x)        % f(x)对x积分
int(f,a,b)      % f的定积分，上限a下限b
diff(f,x,n)     % f(x)关于x的n次导，没有n时默认为1
limit(e,v,a,)   % e式在v->a的值,后跟正极限和负极限
taylor(f,a,n)   % f(x)在a上进行泰勒展开，n为展开阶数
D2y             % 对y微分2次
dsolve(f)       % 求解常微分方程
                % f的例子:'Dx1=x1+3*x2,Dx2=5*x1+x2'
od45(fun,ran,y0)% 求解微分方程的函数,返回时间点向量和对应解
```
四舍五入
```m
floor(x)        % 大于等于的整数
ceil(x)         % 小于等于的整数
round(x)        % 最接近的整数
fix(x)          % 取为0
```
进制转换
```
bin:0-1
dec:0-9
base(octal):0-7
hex:0-15 or 0-f
2:to
```
矩阵操作
```m
A=[1  2  3;  4  5  6;  7  9  2];  
A' '            % (多一个'号，阻止渲染错误)，对调行和列
A(:,1) = B;     % 设置第一列为B矩阵
A(2,1)          % 获取第二列第一行的元素
zeros(2,3)      % 2列3行全0
ones(2,3)       % 2列3行全1
[]              % 空矩阵，可用于删除
det(A)          % 计算矩阵对应行列式
diag([1 2 3])   % 对角矩阵，对角线上赋值
                % 如果赋值全为1则为单位矩阵，记作I
rank(A)         % 矩阵的秩，代表维度
inv(A)          % A的逆矩阵：AB = BA = I
                % 对角矩阵的逆矩阵为为的倒数
trace(A)        % 矩阵的迹：对角线元素之和
eig(A)          % 矩阵的特征值
[V,D] = eig(A)  % 矩阵的特征值D与特征向量V
                % 对于一个矩阵变换，特征向量的方向不会改变，长度比例为特征值
                % 线性代数笑话一则 >> 什么是线性代数？... 什么不是线性代数：高等数学出版社-线性代数与解析几何，同济大学数学系-线性代数
x=A\b           % Ax = b的高效计算    
any             % 检查是否有非0
all             % 检查是否全是非0
```

信号操作
```m
conv(x,h)       % 卷积，x和h均为有限序列
flplr           % 翻转    
tf(i,o)         % 创建一个变换系统,i分子,o分母
num(a)          % 创建分子多项式,接受数组
den(a)          % 创建分母多项式,同上
zpk(z,p,k)      % 创建一个零极点增益$\frac{k(s+z)}{(s+p)}$,参数均可接受数组
zplane(z,p)     % 绘制零极点,zp可以是数组,也可以是系统和输入
fourier(f,t,w)  % 对f进行傅里叶变换，将t（代表时间的变量）转为w
ifourier(F,w,t) % 对f进行傅里叶逆变换，同上
laplace(f,t,s)  % 拉普拉斯变换
                % 拉氏变换就是对信号乘以e_{-at}次方的傅里叶变换，以保证信号绝对可积，其中a的参数需要自行调整至合适程度

[N,Wc]=buttord(Wp,Ws,Rp,Rs,’s’); 
      =cheb1ord
      =cheb2ord
      =ellipord
% 四类滤波器,超值黄油,切比雪夫I/II型,椭圆
% 参数意义:滤波器阶数,满足要求的截止频率,通带边界频率,阻带边界频率,通带允许的最大衰减,阻带需要达到的最小衰减,s连续z数字
[z,p,k]=buttap(n);  
       =cheb1ap(N,Rp); 
       =cheb2ap(N,Rs); 
       =ellipap(N,Rp,Rs); 
% 归一化滤波器
% 参数意义:零点数组,极点数组,增益
zp2tf(z,p,k)    % 将零点数组,极点数组,增益转为有理函数形式

[num,den]=lp2lp(num,den,Wc);  
[num,den]=lp2hp(num,den,Wc);  
[num,den]=lp2bp(num,den,W0,Bw);  
[num,den]=lp2bs(num,den,W0,Bw);   
% lp/hp/hp/bs:低通/高通/带通/带阻
% wc:截止频率,w0中心频率,bw带宽

[num,den]=cheby1(N, Rp,Wc,’s’);  
[num,den]=cheby1(N,Rp,Wc,’high’,’s’); 
[num,den]=cheby1(N,Rp,Wc,’stop’,’s’); 
[num,den]=cheby1(N,Rp,Wc,’s’);  
[z,p,k]=cheby1(N,Rp,Wc,’s’);  
[z,p,k]=cheby1(N,Rp,Wc,’high’,’s’);  
[z,p,k]=cheby1(N,Rp,Wc,’stop’,’s’);  
[z,p,k]=cheby1(N,Rp,Wc,’s’); 
```

画图操作
```m
stem(x,y)       % 绘制x与y的图像
grid            % 切换图像网格
ezplot(f,[-1,1])% 绘制f在[-1,1]上的图像
ezsurf(f,domain)% 绘制彩色3D图像
legend          % 为图像添加标识
polyfit(x,y,3)  % 已知一组x,y的数据,用3次多项式拟合该离散点
interp1(x,y,xq) % 已知数据x,y,估算未知点x,称为插值
                % 参数:linear线性,nearest最近,spline平滑,pchip分段三次 Hermite 插值
```