
![Logo](https://user-images.githubusercontent.com/59482214/170513866-e6293384-6398-4418-b6e8-79388af5c852.png)


# وارق الذكي 

نظام وارق الذكي مدرب على تصنيف 100 نوع من النباتات التي تنمو في بيئة المملكة العربية السعودية . 
تم تدريب وارق باستخدام مجموعة بيانات مسخرجة من دليل نباتات منطقة الرياض. 
## المصادر المستخدمة لاستخراج البيانات :
 - [دليل نباتات الرياض ](https://www.riyadhenv.gov.sa/wp-content/uploads/2019/10/Riyadh-Plants-Manual-Ar.pdf)
 
## واجهة المستخدم : 


<img width="935" alt="ui" src="https://user-images.githubusercontent.com/59482214/170519596-f0808c98-4ff0-4d5e-adb6-fff738c36015.png">
<img width="857" alt="uir" src="https://user-images.githubusercontent.com/59482214/170519661-b3e6bcae-03b1-4da3-b361-3d410b2b9ab3.png">



## طريقة الاستخدام  : 
1. قم بتحميل صورة 
2. اضغط على زر ابدأ
3. سوف تظهر نتيجة التنبؤ باسم النبتة و موعد نموها و نوعها و لون زهرتها 


## نشر التطبيق

To deploy this project run

```bash
  npm run python app.py 
```




## معلومات النموذج 

| القيمة           | الاسم                                                               |
| ----------------- | ------------------------------------------------------------------ |
| keras Sequential Model | النوع |
| VGG16 | النموذج الاساسي  |
| 97% |الدقة |
| التصنيف|الوظيفة |
| 0.13 | نسبة الخطأ |
| categorical_crossentropy | معادلة الخسارة  |
|accuracy | metrics  |
|39,657 | حجم البيانات |
|90% | نسبة التدريب   |
|10% |نسبة الاختبار   |






<img width="184" alt="Screenshot 2022-05-26 181528" src="https://user-images.githubusercontent.com/59482214/170518638-eca3e99b-75a4-4131-94e1-6a8ee460188c.png">



## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/OhoodAljohani/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)]([https://www.linkedin.com/](https://www.linkedin.com/in/ohood-aljohani-415719202/))
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)]([https://twitter.com/](https://twitter.com/UdiO13))

