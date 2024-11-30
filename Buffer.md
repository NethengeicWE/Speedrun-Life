## 不是，哥们，你认真的？
* 习近平新时代中国特色社会主义思想的历史地位
    1. 习近平新时代中国特色社会主义思想是当代中国马克恩主义、二十一世纪马克思主是中华文化和中国精神的时代精华，实现了马克恩主义中国化时代化新的飞跃。
    1. 继承和发展马克思列宁主义毛泽东思想、邓小平理论、“三个代表”重要思想、科学
    发展观，是马克思主义在当代中国发展的最新理论成果，开踪了马克思主义中国化时代化都
    境界。
    1. 把马克思主义基本原理同中国具体实际相结合、同中华优秀传统文化相结合，使马
    思主义这个魂脉和中华优秀传统文化这个根账内在贯通、相互成就，是中华民族的文化主供
    生最有力的体现，是中华文化和中国精神的时代精华
    1. 是全党全军全国各族人民为实现中华民族伟大复兴的行动指南，是新时代党和国家事业的根本遵循
 *





## 实验课慢慢改
```
CREATE (n:Person {name:'菩提祖师'}) RETURN n;
CREATE (n:Person {name:'唐玄奘'}) RETURN n;
CREATE (n:Person {name:'猪八戒'}) RETURN n;
CREATE (n:Person {name:'沙悟净'}) RETURN n;
CREATE (n:Person {name:'孙悟空'}) RETURN n;
CREATE (n:Person {name:'白龙马'}) RETURN n;
CREATE (n:Person {name:'西海龙王'}) RETURN n;
CREATE (n:Person {name:'如来佛祖'}) RETURN n;
CREATE (n:Person {name:'观音菩萨'}) RETURN n;
CREATE (n:Person {name:'金鼻白毛飘精'}) RETURN n;
CREATE (n:Person {name:'李靖'}) RETURN n;
CREATE (n:Person {name:'牛魔王'}) RETURN n;
CREATE (n:Person {name:'铁扇公主'}) RETURN n;
CREATE (n:Person {name:'红孩儿'}) RETURN n;
CREATE (n:Person {name:'哪吒'}) RETURN n;
CREATE (n:Person {name:'木吒'}) RETURN n;
CREATE (n:Person {name:'金吒'}) RETURN n;
CREATE (n:Person {name:'太乙真人'}) RETURN n;
CREATE (n:Person {name:'鸿钧老祖'}) RETURN n;
CREATE (n:Person {name:'太上老君'}) RETURN n;
CREATE (n:Person {name:'灵宝天尊'}) RETURN n;
CREATE (n:Person {name:'元始天尊'}) RETURN n;
CREATE (n:Person {name:'王母娘娘'}) RETURN n;
CREATE (n:Person {name:'云中子'}) RETURN n;
CREATE (n:Person {name:'玉鼎真人'}) RETURN n;
CREATE (n:Person {name:'姜子牙'}) RETURN n;
CREATE (n:Person {name:'申公豹'}) RETURN n;
CREATE (n:Person {name:'雷震子'}) RETURN n;
CREATE (n:Person {name:'杨戬'}) RETURN n;
CREATE (n:Person {name:'玉皇大帝'}) RETURN n;
CREATE (n:Person {name:'沉香'}) RETURN n;
CREATE (n:Person {name:'三圣母'}) RETURN n;

MATCH (a:Person {name:'唐玄奘'}), (b:Person {name:'猪八戒'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'唐玄奘'}), (b:Person {name:'沙悟净'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'唐玄奘'}), (b:Person {name:'白龙马'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'孙悟空'}), (b:Person {name:'唐玄奘'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'鸿钧老祖'}), (b:Person {name:'太上老君'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'鸿钧老祖'}), (b:Person {name:'灵宝天尊'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'鸿钧老祖'}), (b:Person {name:'元始天尊'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'元始天尊'}), (b:Person {name:'太乙真人'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'元始天尊'}), (b:Person {name:'云中子'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'元始天尊'}), (b:Person {name:'玉鼎真人'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'元始天尊'}), (b:Person {name:'姜子牙'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'元始天尊'}), (b:Person {name:'申公豹'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'云中子'}), (b:Person {name:'雷震子'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'玉鼎真人'}), (b:Person {name:'杨戬'}) MERGE (a)-[:师徒]->(b);
MATCH (a:Person {name:'哪吒'}), (b:Person {name:'太乙真人'}) MERGE (a)-[:师徒]->(b);

MATCH (a:Person {name:'西海龙王'}), (b:Person {name:'白龙马'}) MERGE (a)-[:父子]->(b);
MATCH (a:Person {name:'牛魔王'}), (b:Person {name:'红孩儿'}) MERGE (a)-[:父子]->(b);
MATCH (a:Person {name:'李靖'}), (b:Person {name:'金吒'}) MERGE (a)-[:父子]->(b);
MATCH (a:Person {name:'李靖'}), (b:Person {name:'木吒'}) MERGE (a)-[:父子]->(b);
MATCH (a:Person {name:'李靖'}), (b:Person {name:'哪吒'}) MERGE (a)-[:父子]->(b);

MATCH (a:Person {name:'孙悟空'}), (b:Person {name:'菩提祖师'}) MERGE (a)-[:第一任师傅逆天技能传授]->(b);
MATCH (a:Person {name:'孙悟空'}), (b:Person {name:'观音菩萨'}) MERGE (a)-[:紧窟咒]->(b);
MATCH (a:Person {name:'孙悟空'}), (b:Person {name:'如来佛祖'}) MERGE (a)-[:五指山]->(b);
MATCH (a:Person {name:'观音菩萨'}), (b:Person {name:'唐玄奘'}) MERGE (a)-[:雇主]->(b);
MATCH (a:Person {name:'如来佛祖'}), (b:Person {name:'金吒'}) MERGE (a)-[:前部护法]->(b);
MATCH (a:Person {name:'观音菩萨'}), (b:Person {name:'木吒'}) MERGE (a)-[:大弟子]->(b);
MATCH (a:Person {name:'观音菩萨'}), (b:Person {name:'红孩儿'}) MERGE (a)-[:坐下善财童子]->(b);
MATCH (a:Person {name:'铁扇公主'}), (b:Person {name:'红孩儿'}) MERGE (a)-[:母子]->(b);
MATCH (a:Person {name:'牛魔王'}), (b:Person {name:'铁扇公主'}) MERGE (a)-[:夫妻]->(b);
MATCH (a:Person {name:'王母娘娘'}), (b:Person {name:'元始天尊'}) MERGE (a)-[:父女]->(b);
MATCH (a:Person {name:'杨戬'}), (b:Person {name:'三圣母'}) MERGE (a)-[:兄妹]->(b);
MATCH (a:Person {name:'三圣母'}), (b:Person {name:'沉香'}) MERGE (a)-[:]->(b);
MATCH (a:Person {name:'玉皇大帝'}), (b:Person {name:'杨戬'}) MERGE (a)-[:母子]->(b);
MATCH (a:Person {name:'唐玄奘'}), (b:Person {name:'金鼻白毛飘精'}) MERGE (a)-[:过家家]->(b);
MATCH (a:Person {name:'金鼻白毛飘精'}), (b:Person {name:'李靖'}) MERGE (a)-[:义女]->(b);


```

## 最怕的是水课老师觉得自己不水
```
第1题   您的性别      [单选题]
男	36	  72%
女	14	  28%
本题有效填写人次	50	

第2题   您所在的年级      [单选题]
选项	小计	比例
大一	5	  10%
大二	5	  10%
大三	37	  74%
大四	3	  6%
本题有效填写人次	50	

第3题   平均每个月生活费      [单选题]
选项	小计	比例
1000以下	10	  20%
1000~2000	31	  62%
2000以上	9	  18%
本题有效填写人次	50	

第4题   每天吃饭会花多少钱      [单选题]
选项	小计	比例
10~20	11	  22%
20~30	26	  52%
30以上	13	  26%
本题有效填写人次	50	

第5题   父母给的生活费还满意吗？      [单选题]
选项	小计	比例
很满意	29	  58%
一般	19	  38%
不满意	2	  4%
本题有效填写人次	50	

第6题   生活费给的频率      [单选题]
选项	小计	比例
一月一次	26	  52%
一学期一次	6	  12%
没了就给	18	  36%
本题有效填写人次	50	

第7题   父母会刻意控制你的生活费吗？[单选题]
选项	小计	比例
不会	39	  78%
偶尔	9	  18%
经常	2	  4%
本题有效填写人次	50	

第8题   生活费的来源？[单选题]
选项	小计	比例
父母提供	43	  86%
兼职	5	  10%
奖学金	2	  4%
本题有效填写人次	50	

第9题   生活费不够的时候通过哪种方式获得      [单选题]
选项	小计	比例
父母提供	27	  54%
兼职	17	  34%
借款/花呗	6	  12%
本题有效填写人次	50	

第10题   你平时会有意识的攒钱吗? [单选题]
选项	小计	比例
会	35	  70%
不会	15	  30%
本题有效填写人次	50	

第11题   会出现月光吗      [单选题]
选项	小计	比例
不会	22	  44%
偶尔	16	  32%
经常	12	  24%
本题有效填写人次	50	

第12题   有谈恋爱吗？      [单选题]
选项	小计	比例
正在恋爱	20	  40%
恋爱过	7	  14%
没有恋爱过	23	  46%
本题有效填写人次	50	

第13题   谈恋爱后父母给的生活费增加了多少？      [单选题]
选项	小计	比例
没有	34	  68%
0~1000	9	  18%
1000~2000	2	  4%
2000以上	5	  10%
本题有效填写人次	50	

第14题   每月消费最多的是什么？      [多选题]
选项	小计	比例
吃	47	  94%
娱乐活动(聚餐 KTV 电影院等)	19	  38%
日常用品	22	  44%
化妆品	9	  18%
其他	7	  14%
本题有效填写人次	50	
```
,这个调查是为了完成《写作与沟通》课程的调研报

为了完成《写作与沟通》课程要求的调研报告，我们发布了大学生生活费情况调查并获取了调查数据。请生成该报告的大纲。