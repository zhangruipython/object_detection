# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-03 14:22
@Author  : zhangrui
@FileName: demo_service.py
@Software: PyCharm
"""
from new_project.demo.models import Publication, Article

p1 = Publication(title="pub01")
p1.save()
p2 = Publication(title="pub02")
p2.save()
a1 = Article(headline="arc01")
a2 = Article(headline="arc02")
a1.save()
a2.save()
a1.publication.add(p1, p2)
