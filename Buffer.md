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