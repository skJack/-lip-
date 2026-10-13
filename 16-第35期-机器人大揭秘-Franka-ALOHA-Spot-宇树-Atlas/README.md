# 第 35 期 · 具身智能基础（三）

**视频标题**：机器人大揭秘

论文和机器人视频里天天出镜的，其实就这十几台。这期按五类本体一台一台认过去，每台讲清楚谁家做的、什么来头、基本参数、能干什么、值多少钱：**单臂**（Franka FR3、UR5e、SO-101，再带 WidowX、YAM、ARX / PiPER）、**手**（二指夹爪；灵巧手 Shadow、Sharpa、因时，2019 年转魔方对比 2026 年削苹果；再带 LEAP、Allegro、NEO 和 Atlas 的手）、**双臂**（ALOHA 和 Mobile ALOHA / ALOHA 2、智元精灵，再带星海图 R1、Stretch）、**四足**（Spot、宇树 Go2，再带 ANYmal、云深处山猫）、**人形**（宇树 G1、Atlas、Optimus、Figure 03，再带 Digit、NEO、天工、Walker S2、小鹏 IRON）。开场有一条从 1961 年 Unimate 到 2026 年的时间线、一页 13 家公司一览、一张按真实比例站成一排的全家福；最后两张对照表和一条从两百多美元到十万美元以上的价格轴。数字都是 2026 年 10 月初查的。

## 这期的材料从哪来

这期没有论文，讲的是机器人本身。参数、价格和公司现状来自官网、数据表、交易所文件和媒体报道，逐条出处在 `notes/技术事实核查.md`。deck 里的 35 段演示视频都是从下面这些官方视频里切的几秒到二十几秒；仓库里不放视频文件，视频页换成了片段的第一帧，想看动的点下面的链接。机器人的 3D 图（全家福、夹爪、ALOHA、Spot、Go2、G1）是用 MuJoCo Menagerie 的公开模型自己渲染的。

| 类型 | 材料 | 链接 |
|---|---|---|
| 模型 | MuJoCo Menagerie（全家福和几张 3D 图用的公开机器人模型） | <https://github.com/google-deepmind/mujoco_menagerie> |
| 第 6 页 | Franka Robotics《Force motion》 | <https://www.youtube.com/watch?v=ClsPINlOTwg> |
| 第 7 页 | UMI 项目页「Ours (UR5)」 | <https://umi-gripper.github.io/> |
| 第 8 页 | Hugging Face《What can you build during 48h Robotics Hackathon?》 | <https://www.youtube.com/watch?v=FLxFBU-Kodk> |
| 第 10 页 | OpenVLA 项目页（WidowX 叠杯） | <https://openvla.github.io/> |
| 第 10 页 | ABC 项目页（YAM 装 AirPods） | <https://abc.bot/> |
| 第 10 页 | 方舟无限《ARX5 x ALOHA 数据采集测试》 | <https://www.youtube.com/watch?v=wHBDZ7RnRa4> |
| 第 12–13 页 | OpenAI《Solving Rubik's Cube with a Robot Hand》 | <https://www.youtube.com/watch?v=x4O8pojMF0w> |
| 第 12–13 页 | Sharpa《Bimanual Robotic Dexterity: Apple Peeling Demo with Sharpa Wave》 | <https://www.youtube.com/watch?v=UVQdqf8ppuw> |
| 第 12 页 | 因时机器人《仿人五指灵巧手产品介绍》 | <https://www.bilibili.com/video/av113434767131001> |
| 第 14 页 | LEAP Hand（RSS 2023） | <https://www.youtube.com/watch?v=VKAlJPypQFI> |
| 第 14 页 | Wonik《Allegro Hand V4 Introduction》 | <https://www.youtube.com/watch?v=pphnYkghg0A> |
| 第 14 页 | 1X《NEO's Hands》 | <https://www.youtube.com/watch?v=QRyXV3csReA> |
| 第 14 页 | Boston Dynamics《New Hands for Atlas》 | <https://www.youtube.com/watch?v=4whgw2gLBS8> |
| 第 16 页 | Mobile ALOHA: Your Housekeeping Robot | <https://www.youtube.com/watch?v=HaaZ8ss-HP4> |
| 第 16 页 | Google DeepMind《Gemini Robotics Demo: Lunch Packing with ALOHA 2》 | <https://www.youtube.com/watch?v=m-G4-slYcGE> |
| 第 17–18 页 | 智元《Introducing AgiBot World》 | <https://www.youtube.com/watch?v=4Ss3sfUjqTM> |
| 第 18 页 | 智元《AGIBOT G2》 | <https://www.youtube.com/watch?v=PPlnLFLJWZ4> |
| 第 19 页 | 星海图《Stanford x Galaxea … Full-Body Mobile Manipulation Robot Suite》 | <https://www.youtube.com/watch?v=cPqWno2G0XM> |
| 第 19 页 | Hello Robot《Stretch 4 Launch》 | <https://www.youtube.com/watch?v=uyHa-Gk4THw> |
| 第 21 页 | Boston Dynamics《Testing Robustness》（2018，Spot 开门） | <https://www.youtube.com/watch?v=aFuA50H9uek> |
| 第 21 页 | Boston Dynamics《Air Spot · RL Behavior Research》 | <https://www.youtube.com/watch?v=LMPxtcEgtds> |
| 第 23 页 | 宇树《Introducing Unitree Go2》 | <https://www.youtube.com/watch?v=6zPvT0ig1VM> |
| 第 24 页 | ETH《ANYmal Parkour》 | <https://www.youtube.com/watch?v=PjWvf90l4cg> |
| 第 24 页 | 云深处《LYNX M20 Launch》 | <https://www.youtube.com/watch?v=NAcanWv_2Z8> |
| 第 26 页 | 宇树《全球首次完成侧空翻的人形机器人：宇树G1》 | <https://www.bilibili.com/video/BV1v8QdYHEB4> |
| 第 26 页 | 央视新闻《总台马年春晚〈武BOT〉》 | <https://www.bilibili.com/video/BV13mZuBsEao> |
| 第 27–28 页 | Boston Dynamics《Inside the Robotics Race》（现代汽车工厂里的 Atlas） | <https://www.youtube.com/watch?v=CFS_zU1kWQw> |
| 第 28 页 | Boston Dynamics《Atlas · Partners in Parkour》 | <https://www.youtube.com/watch?v=tF4DML7FIWk> |
| 第 29 页 | Tesla《Optimus - Gen 2》 | <https://www.youtube.com/watch?v=cpraXaw7dyc> |
| 第 30–31 页 | Figure《Introducing Figure 03》 | <https://www.youtube.com/watch?v=Eu5mYMavctM> |
| 第 31 页 | Figure《Helix 2.5 30-Home Generalization》 | <https://www.youtube.com/watch?v=lJpM_2a1zrE> |
| 第 32 页 | Agility《Digit's First Day of Work at GXO》 | <https://www.youtube.com/watch?v=AJpTpUqjgrY> |
| 第 32 页 | 1X《NEO The Home Robot · Order Today》 | <https://www.youtube.com/watch?v=LTYMWadOW7c> |
| 第 33 页 | CCTV+《Tiangong Ultra Robot Wins World's First Humanoid Half-Marathon》 | <https://www.youtube.com/watch?v=BYmCxYTbg50> |
| 第 33 页 | 优必选《Walker S2 全球首个实现自主换电》 | <https://www.bilibili.com/video/BV1RDgAz3Esu> |
| 第 33 页 | 小鹏《Cutting Open IRON》 | <https://www.youtube.com/watch?v=m_Ag_SgsHVg> |
| 第 2 页 | 时间线小图：The Henry Ford（Unimate） | <https://www.youtube.com/watch?v=-Xl2c91pWGc> |
| 第 2 页 | 时间线小图：The Telegraph（ASIMO） | <https://www.youtube.com/watch?v=eU93VmFyZbg> |
| 第 2 页 | 时间线小图：Boston Dynamics（BigDog） | <https://www.youtube.com/watch?v=cNZPRsrwumQ> |
| 第 2 页 | 时间线小图：Willow Garage（PR2） | <https://www.youtube.com/watch?v=c3Cq0sy4TBs> |
| 第 2 页 | 时间线小图：Boston Dynamics（Spot 开卖） | <https://www.youtube.com/watch?v=wlkCQXHEgjA> |

## 笔记

- [`notes/技术事实核查.md`](notes/技术事实核查.md)：公司背景（成立年份、创始人、现在归谁、上没上市）和每页数字的出处，每条标「已核 / 媒体 / 口径」三种状态；最后一节是页面上刻意没放的（几个对不上的数，比如天工百米的三个成绩）。

## Slides

- [`slides/deck.html`](slides/deck.html)：视频里用的 PPT，HTML 格式，下载后用 Chrome 打开即可，方向键翻页（快捷键和每页讲什么见 [`slides/使用说明.md`](slides/使用说明.md)）
- [`slides/预览-全部页面.png`](slides/预览-全部页面.png)：全部页面缩略图，不想下载时先看这张

> 这期 deck 里有 35 段演示视频，都是从各公司和项目的官方视频里切的，仓库里不放（近 200 MB），视频页换成了片段的第一帧，原片链接见上面的表。
