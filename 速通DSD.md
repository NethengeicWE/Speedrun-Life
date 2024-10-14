## DSD
### 英语6级
* Mux，Multiplexer：多路复用器
* Tristate Buffer：三态缓冲器:高电平输入输出打通，低电平处于可被覆盖的未知输出
* Sequential：时序，同步
* Asequential：时序，异步
* Semiconductor：半导体
* wordline：地址线
* bitline：数据线
* Field Programmable Gate Array：场可编程门阵列
###  备注
1. 门符号
    * 与门：输入端直线
    * 或门：输入端内凹
    * 非门：三角形
    * 异或门：输入端双内凹
1. 德摩根规则：布尔运算中拆分非门的括号时，括号内的 与 和 或 互换
    * ！(A+B) = !A * !B
    * ! (A*B) = !A + !B
1. 真值表转表达式-KarMap：
    * X表示0/1，代表该输入不影响最终结果
    * 当输出为1时将当前门状态加入表达式
    * 快速表示：复合真值表（复合输入/输出）中以1*2，2*1，2*2形式出现的1可以；  
    可以通过调整表格形式引入更多
1. Bistable，D Latch：同一个东西，两个非门首尾相连
1. SR Latch：置位/复位锁存器，两个或非门，输出接入另一个逻辑门的输入，Set针脚一侧输出非Q，Reset针脚输出Q；S0R1输出Q0，S1R0输出Q1，S1R1不存在，S0R0保持输出
    * D Latch：SR Latch的扩展，将输入换为D与clk与门的输出，使得clk为高电平时使能D
    * DFF：两个D Latch首尾相连，Data输入的clk输入接上非门，Q输出的clk接口保持不变，这样就能实现在上升沿使能
1. 有限状态机Finite State Machine
    * 步骤：定义输入输出信号，确定状态，确定转换条件，确定输出逻辑
    * Moore Machine：输出取决于当前状态
    * Mealy Machine：输出取决于当前状态与输入，状态数量更少
1. Mutiplexer：利用TristateBuffer实现
1. Shifer 移位寄存器：利用串联的dff实现多根信号与单根线信号的转换（空间换时间）
1. 超前进位加法器
    * 优点：有且只有4逻辑门的延迟
    * 缺点：体积大
    * 设$G_{i}=A_{i}·B_{i},P_{i}=A_{i} xor B_{i}$
        * 有$C_{i+1} = G_{i} + P_{i}·C{i}$,递归该式子即可得到公式
    * 模块打包
        * 1bit全加模块：  
        S = (A xor B) xor C  
        p = A xor B  
        G = A and B
        * nbit进位模块：包含所有参与位的操作  
        $C_{out} = G_{n:0}+P_{n:0}·C_{in}$  
        $P_{n:0} = P_{0} + ... + P_{n}$  
        $G_{i+1} = G_{i}+P{i}·$ 


1. VHDL语法
    ```VHDL
    -- => 连接两个端口
    -- =: 赋值
    -- = 比较

    ibrary IEEE
    use ieee.std_logic_1164.all
    entity decoder is 
        port {
            a: in std_logic_vector(1 downto 0);
            b: out std_logic_vector(3 downto 0);
        }
    end decoder;

    architecture rtl of decoder is 
    begin 
        -- signal tmp:std_logic;
            -- physical architecture signal wire
        -- variable tmp0:std_logic;
            -- for variable,using =: to assignment
        process (a)
        begin
            case a is
                when "00" => b <= "0001";
                when "01" => b <= "0010";
                when "10" => b <= "0100";
                when "11" => b <= "1000";
            end case;
        end process;
    end rtl;
    ```
    定义数据类型
    ```VHDL
    architecture Behavioral of fsmdesign is 
        type state_type is (s_waitcard,s_waitpass);
        signal state,next_state: state_type;
        begin 
            -- etc
     end rtl;
    ```
    时序控制：以计数器为例
    ```VHDL
    architecture rtl of bcounter is 
    signal cnt: unsigned(31 downto 0);
    begin
        q <= std_logic_vector(cnt);
        proc_cnt:process(clk);
        begin
            if (clk'event and clk='1') then
                cnt <= cnt + 1;
            end if;
        end process;
    end rtl;
    ```
    带有异步复位的DFF，随时重置数值  
    同步复位把clr信号处理写入clk'event中
    ```VHDL
    architecture rtl of foo is begin
        proc_simprefclr: process (clk,clr) begin
            if clr = '1' then
                q <= "0";
            elsif clk'event and clk = '1' then
                q <= d;
            end if
        end process;
    end rtl;
    ```
    ROM
    ```VHDL
    use ieee.numberic_std.all;
    acchitecture rtl of myrom is 
        type romtype is array(0,15) of std_logic_vector(7 downto 0);
        constant romdata : romtype := 
            ("11001010","10101010",others => "0000000");
        begin
            process(adder) begin 
                dout <= romdata (to_interger(unsigned(addr)))
            end process;
    end rtl
    ```
    ```VHDL
    library IEEE;
    use IEEE.STD_LOGIC_1164.all;
    entity decoder_5 is
    port(
        a : in STD_LOGIC_VECTOR(3 downto 0);
        b : out STD_LOGIC_VECTOR(6 downto 0)
    );
    end decoder_5;

    architecture bhv of decoder_5 is
        begin
        process(a) begin
                case a is
                    when "0000" => b <= "1000000"; -- 0
                    when "0001" => b <= "1111001"; -- 1
                    when "0010" => b <= "0100100"; -- 2
                    when "0011" => b <= "0110000"; -- 3
                    when "0100" => b <= "0011001"; -- 4
                    when "0101" => b <= "0010010"; -- 5
                    when "0110" => b <= "0000010"; -- 6
                    when "0111" => b <= "1111000"; -- 7 
                    when "1000" => b <= "0000000"; -- 8 
                    when "1001" => b <= "0010000"; -- 9 
                    when "1010" => b <= "0001000"; -- A 
                    when "1011" => b <= "0000011"; -- b 
                    when "1100" => b <= "1000110"; -- C 
                    when "1101" => b <= "1000001"; -- d
                    when "1110" => b <= "0000110"; -- E 
                    when "1111" => b <= "0001110"; -- F  
                    when others => b <= "1111111";
                end case;
        end process;
    end bhv;

    ```