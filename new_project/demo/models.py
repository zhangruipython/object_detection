from django.db import models


# 多对多关系 通过ManyToManyField创建中间表
class Publication(models.Model):
    title = models.CharField(max_length=30)

    class Meta:
        ordering = ['title']
        db_table = 'Publication'

    def __str__(self):
        return self.title


class Article(models.Model):
    headline = models.CharField(max_length=100)
    publication = models.ManyToManyField(Publication)

    class Meta:
        ordering = ['headline']
        db_table = 'Article'

    def __str__(self):
        return self.headline


# 多对一关系 通过外键关联 ForeignKey

class Reporter(models.Model):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)

    class Meta:
        db_table = 'Report'


class Text(models.Model):
    headline = models.CharField(max_length=30)
    reporter = models.ForeignKey(Reporter, on_delete=models.CASCADE)

    class Meta:
        db_table = 'Text'


# 创建一对一关系表 例如一家餐馆对应唯一的地址，而餐馆和地址是不同的对象有不同的属性所以需要进行双表一对一关联
class Place(models.Model):
    name = models.CharField(max_length=30)
    address = models.CharField(max_length=60)


class Restaurant(models.Model):
    place = models.OneToOneField(Place, on_delete=models.CASCADE, primary_key=True)
    pizza = models.BooleanField(default=False)


class Waiter(models.Model):
    restaurant = models.ForeignKey(Restaurant, on_delete=models.CASCADE)
    name = models.CharField(max_length=30)
