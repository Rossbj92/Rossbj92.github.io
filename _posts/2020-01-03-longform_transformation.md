---
title: "Inclusion of the Brand in Self (IBIS)"
date: 2020-01-03
tags: [data wrangling, longform]
header:
  image: "/images/diary.jpg"
excerpt: ""
mathjax: "true"
---

This collection of notebooks surrounds a diary study that I assisted with. Diary studies are longitudinal studies in which participants routinely complete surveys throughout. These are especially useful for tracking things such as mood, exercise/diet habits, and for our intents and purposes, social media usage. 

In this notebook, I'll be walking through the process of converting the raw data from wide to long format. All of the accompanying datasets can be found here [insert link].

In the initial survey (day 0), participants were asked to give several different types of ratings for 6 social media platforms: Facebook, Youtube, Snapchat, Instagram, Twitter, and Pinterest. Aside from the IBIS measure, several ubiquitous market research measures were used:
1. Purchase Intention ("_use")
2. Overall opinion ("_op")
3. Likelihood to recomment ("_rec")

The IBIS measure is adapted from the Inclusion of Others in Self (IOS) scale, originally developed as a way to measure to measure intimacy. Individuals are simply asked about their relationship with a partner and shown a series of increasingly overlapping circles - the idea being that more overlap will indicate stronger relationships.
<img src="/images/ios.png" alt="linearly separable data">
Adaptation of this measure was originally proposed following the rise in the theoretical perspective of brand relationships, or the notion that individuals develop relationships with brands similarly to ones with other other people. The bulk of the work concerning this measure had been mostly conceptual, and to date, our study would be the first using actual usage data.

We had 2 main questions:
1. Does IBIS prospectively predict usage?
2. If so, how does it perform relative to other widely used measures?

# Day 0 Transformation

First, we'll need to import the libraries that we will need. The "set_option" parameters are there to make viewing all of the columns easier.

There were several components to the overall experiment, and this current cleaning/analysis will focus on social media platforms.

```python
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
```

Let's take a look at the data.

```python
day_zero = pd.read_csv('day_zero.csv')

day_zero.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>age</th>
      <th>ethnicity</th>
      <th>gender</th>
      <th>fsc_1</th>
      <th>fsc_2</th>
      <th>fsc_3</th>
      <th>fsc_4</th>
      <th>aquafina_percep</th>
      <th>dasani_percep</th>
      <th>fiji_percep</th>
      <th>starbucks_percep</th>
      <th>coffee_bean_percep</th>
      <th>dunkin_donuts_percep</th>
      <th>freudian_sip_percep</th>
      <th>twitter_percep</th>
      <th>snapchat_percep</th>
      <th>youtube_percep</th>
      <th>coke_percep</th>
      <th>sprite_percep</th>
      <th>pepsi_percep</th>
      <th>budweiser_percep</th>
      <th>miller_lite_percep</th>
      <th>corona_percep</th>
      <th>blue_moon_percep</th>
      <th>coors_percep</th>
      <th>facebook_percep</th>
      <th>instagram_percep</th>
      <th>pinterest_percep</th>
      <th>coke_percep.1</th>
      <th>sprite_percep.1</th>
      <th>pepsi_percep.1</th>
      <th>budweiser_percep.1</th>
      <th>miller_lite_percep.1</th>
      <th>corona_percep.1</th>
      <th>blue_moon_percep.1</th>
      <th>coors_percep.1</th>
      <th>facebook_percep.1</th>
      <th>instagram_percep.1</th>
      <th>pinterest_percep.1</th>
      <th>aquafina_percep.1</th>
      <th>dasani_percep.1</th>
      <th>fiji_percep.1</th>
      <th>starbucks_percep.1</th>
      <th>coffee_bean_percep.1</th>
      <th>dunkin_donuts_percep.1</th>
      <th>freudian_sip_percep.1</th>
      <th>twitter_percep.1</th>
      <th>snapchat_percep.1</th>
      <th>youtube_percep.1</th>
      <th>corona_buy</th>
      <th>blue_moon_buy</th>
      <th>aquafina_buy</th>
      <th>dasani_buy</th>
      <th>fiji_buy</th>
      <th>starbucks_buy</th>
      <th>coffee_bean_buy</th>
      <th>dunkin_donuts_buy</th>
      <th>freudian_sip_buy</th>
      <th>coke_buy</th>
      <th>sprite_buy</th>
      <th>pepsi_buy</th>
      <th>budweiser_buy</th>
      <th>coors_buy</th>
      <th>miller_lite_buy</th>
      <th>coke_buy.1</th>
      <th>sprite_buy.1</th>
      <th>pepsi_buy.1</th>
      <th>budweiser_buy.1</th>
      <th>coors_buy.1</th>
      <th>miller_lite_buy.1</th>
      <th>corona_buy.1</th>
      <th>blue_moon_buy.1</th>
      <th>aquafina_buy.1</th>
      <th>dasani_buy.1</th>
      <th>fiji_buy.1</th>
      <th>starbucks_buy.1</th>
      <th>coffee_bean_buy.1</th>
      <th>dunkin_donuts_buy.1</th>
      <th>freudian_sip_buy.1</th>
      <th>facebook_use</th>
      <th>youtube_use</th>
      <th>snapchat_use</th>
      <th>twitter_use</th>
      <th>instagram_use</th>
      <th>pinterest_use</th>
      <th>pinterest_use.1</th>
      <th>instagram_use.1</th>
      <th>twitter_use.1</th>
      <th>snapchat_use.1</th>
      <th>youtube_use.1</th>
      <th>facebook_use.1</th>
      <th>corona_rec</th>
      <th>blue_moon_rec</th>
      <th>aquafina_rec</th>
      <th>dasani_rec</th>
      <th>fiji_rec</th>
      <th>starbucks_rec</th>
      <th>coffee_bean_rec</th>
      <th>dunkin_donuts_rec</th>
      <th>freudian_sip_rec</th>
      <th>coke_rec</th>
      <th>sprite_rec</th>
      <th>pepsi_rec</th>
      <th>budweiser_rec</th>
      <th>coors_rec</th>
      <th>miller_lite_rec</th>
      <th>facebook_rec</th>
      <th>youtube_rec</th>
      <th>snapchat_rec</th>
      <th>twitter_rec</th>
      <th>instagram_rec</th>
      <th>pinterest_rec</th>
      <th>pinterest_rec.1</th>
      <th>instagram_rec.1</th>
      <th>twitter_rec.1</th>
      <th>snapchat_rec.1</th>
      <th>youtube_rec.1</th>
      <th>facebook_rec.1</th>
      <th>miller_lite_rec.1</th>
      <th>coors_rec.1</th>
      <th>budweiser_rec.1</th>
      <th>pepsi_rec.1</th>
      <th>sprite_rec.1</th>
      <th>coke_rec.1</th>
      <th>freudian_sip_rec.1</th>
      <th>dunkin_donuts_rec.1</th>
      <th>coffee_bean_rec.1</th>
      <th>starbucks_rec.1</th>
      <th>fiji_rec.1</th>
      <th>dasani_rec.1</th>
      <th>aquafina_rec.1</th>
      <th>blue_moon_rec.1</th>
      <th>corona_rec.1</th>
      <th>corono_op</th>
      <th>blue_moon_op</th>
      <th>aquafina_op</th>
      <th>dasani_op</th>
      <th>fiji_op</th>
      <th>starbucks_op</th>
      <th>coffee_bean_op</th>
      <th>dunkin_donuts_op</th>
      <th>freudian_sip_op</th>
      <th>coke_op</th>
      <th>sprite_op</th>
      <th>pepsi_op</th>
      <th>budweiser_op</th>
      <th>coors_op</th>
      <th>miller_lite_op</th>
      <th>facebook_op</th>
      <th>youtube_op</th>
      <th>snapchat_op</th>
      <th>instagram_op</th>
      <th>twitter_op</th>
      <th>pinterest_op</th>
      <th>pinterest_op.1</th>
      <th>instagram_op.1</th>
      <th>twitter_op.1</th>
      <th>snapchat_op.1</th>
      <th>youtube_op.1</th>
      <th>facebook_op.1</th>
      <th>miller_lite_op.1</th>
      <th>coors_op.1</th>
      <th>budweiser_op.1</th>
      <th>pepsi_op.1</th>
      <th>sprite_op.1</th>
      <th>coke_op.1</th>
      <th>freudian_sip_op.1</th>
      <th>dunkin_donuts_op.1</th>
      <th>coffee_bean_op.1</th>
      <th>starbucks_op.1</th>
      <th>fiji_op.1</th>
      <th>dasani_op.1</th>
      <th>aquafina_op.1</th>
      <th>blue_moon_op.1</th>
      <th>corona_op</th>
      <th>pos_1</th>
      <th>pos_2</th>
      <th>pos_3</th>
      <th>pos_4</th>
      <th>pos_5</th>
      <th>neg_1</th>
      <th>neg_2</th>
      <th>neg_3</th>
      <th>neg_4</th>
      <th>neg_5</th>
      <th>neg_6</th>
      <th>neg_7</th>
      <th>neg_8</th>
      <th>participant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>18</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>18</td>
      <td>6</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>18</td>
      <td>5</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7</td>
      <td>6</td>
      <td>9</td>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>18</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3</td>
      <td>6</td>
      <td>8</td>
      <td>6</td>
      <td>7</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

Now, we need to grab the relevant social media columns.

```python
#Getting social media columns and sorting
sm_cols = [col for col in day_zero if 'participant' in col or 'facebook' in col
          or 'youtube' in col or 'snapchat' in col or 'instagram' in col
          or 'twitter' in col or 'pinterest' in col]
sm_cols.sort()

day_zero_sm = day_zero[sm_cols]
day_zero_sm.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>facebook_op</th>
      <th>facebook_op.1</th>
      <th>facebook_percep</th>
      <th>facebook_percep.1</th>
      <th>facebook_rec</th>
      <th>facebook_rec.1</th>
      <th>facebook_use</th>
      <th>facebook_use.1</th>
      <th>instagram_op</th>
      <th>instagram_op.1</th>
      <th>instagram_percep</th>
      <th>instagram_percep.1</th>
      <th>instagram_rec</th>
      <th>instagram_rec.1</th>
      <th>instagram_use</th>
      <th>instagram_use.1</th>
      <th>participant</th>
      <th>pinterest_op</th>
      <th>pinterest_op.1</th>
      <th>pinterest_percep</th>
      <th>pinterest_percep.1</th>
      <th>pinterest_rec</th>
      <th>pinterest_rec.1</th>
      <th>pinterest_use</th>
      <th>pinterest_use.1</th>
      <th>snapchat_op</th>
      <th>snapchat_op.1</th>
      <th>snapchat_percep</th>
      <th>snapchat_percep.1</th>
      <th>snapchat_rec</th>
      <th>snapchat_rec.1</th>
      <th>snapchat_use</th>
      <th>snapchat_use.1</th>
      <th>twitter_op</th>
      <th>twitter_op.1</th>
      <th>twitter_percep</th>
      <th>twitter_percep.1</th>
      <th>twitter_rec</th>
      <th>twitter_rec.1</th>
      <th>twitter_use</th>
      <th>twitter_use.1</th>
      <th>youtube_op</th>
      <th>youtube_op.1</th>
      <th>youtube_percep</th>
      <th>youtube_percep.1</th>
      <th>youtube_rec</th>
      <th>youtube_rec.1</th>
      <th>youtube_use</th>
      <th>youtube_use.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>3</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>4</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

 Brand/platform ratings were counterbalanced, which is why there are so many missing values. To simplify the columns for analysis, I'll simply fill the NAs with 0s and add each respective platform's columns together.

```python
platforms = ['facebook', 'youtube', 'snapchat', 'instagram', 'twitter', 'pinterest']
measures = ['op', 'percep', 'rec', 'use']

for platform in platforms:
    for measure in measures:
        day_zero_sm[f'{platform}_{measure}_final'] = (
        day_zero_sm.filter(like = platform)
         .filter(like = measure)
         .sum(axis = 1)
        )

```

```python
#Filtering and renaming 'final' columns
day_zero_sm_fin = day_zero_sm[[col for col in day_zero_sm.columns if '_final' in col
                              or 'participant' in col]]
day_zero_sm_fin.columns = day_zero_sm_fin.columns.str.rstrip('_final')
```


```python
day_zero_sm_fin.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>participant</th>
      <th>facebook_op</th>
      <th>facebook_percep</th>
      <th>facebook_rec</th>
      <th>facebook_use</th>
      <th>youtube_op</th>
      <th>youtube_percep</th>
      <th>youtube_rec</th>
      <th>youtube_use</th>
      <th>snapchat_op</th>
      <th>snapchat_percep</th>
      <th>snapchat_rec</th>
      <th>snapchat_use</th>
      <th>instagram_op</th>
      <th>instagram_percep</th>
      <th>instagram_rec</th>
      <th>instagram_use</th>
      <th>twitter_op</th>
      <th>twitter_percep</th>
      <th>twitter_rec</th>
      <th>twitter_use</th>
      <th>pinterest_op</th>
      <th>pinterest_percep</th>
      <th>pinterest_rec</th>
      <th>pinterest_use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_dict = {}

for platform in platforms:
    #New df for each platform
    df_dict[platform] = day_zero_sm_fin[['participant',*[col for col in day_zero_sm_fin.columns
                                                     if platform in col]]]
    #Adding in platform name column for eventual long-form
    df_dict[platform]['platform'] = platform
    #Standardizing column names for just measure
    df_dict[platform].columns = [col.split('_')[-1] for col in df_dict[platform].columns]
```


```python
day_zero_master = pd.concat([df_dict[k] for k,v in df_dict.items()])
```

Now that the data are in long-form, each participant should have 6 entries (i.e., 1 entry per platform). Let's do a quick sanity check.

```python
day_zero_master.sort_values('participant').head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>participant</th>
      <th>op</th>
      <th>percep</th>
      <th>rec</th>
      <th>use</th>
      <th>platform</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>facebook</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>snapchat</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>youtube</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>pinterest</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>twitter</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>instagram</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>pinterest</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>facebook</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>snapchat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>instagram</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>youtube</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>twitter</td>
    </tr>
  </tbody>
</table>
</div>



Next, we bring in the weekly survey data. We'll need to make the same long-form transformations before we merge them to the initial survey.

# Weekly Survey Transformations

```python
wk_one = pd.read_csv('week_one.csv')
wk_two = pd.read_csv('week_two.csv')
wk_three= pd.read_csv('week_three.csv')

wk_one.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>coke</th>
      <th>sprite</th>
      <th>pepsi</th>
      <th>other_soda</th>
      <th>bud</th>
      <th>coors</th>
      <th>miller_lite</th>
      <th>corona</th>
      <th>blue_moon</th>
      <th>other_beer</th>
      <th>other_alcohol</th>
      <th>aquafina</th>
      <th>dasani</th>
      <th>fiji</th>
      <th>other_water</th>
      <th>starbucks</th>
      <th>coffee_bean</th>
      <th>dunkin_donuts</th>
      <th>freudian_sip</th>
      <th>other_coffee</th>
      <th>twitter</th>
      <th>snapchat</th>
      <th>youtube</th>
      <th>facebook</th>
      <th>instagram</th>
      <th>pinterest</th>
      <th>participant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1</td>
      <td>162</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>66</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Grabbing social media platforms
wk_dfs = [wk_one, wk_two, wk_three]

for i in range(len(wk_dfs)):
    wk_dfs[i] = wk_dfs[i][[col for col in wk_dfs[i] if 'participant' in col or 'facebook' in col
          or 'youtube' in col or 'snapchat' in col or 'instagram' in col
          or 'twitter' in col or 'pinterest' in col]]

    #Adding in a column for week #
    wk_dfs[i]['week'] = i+1

wk_dfs[0].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitter</th>
      <th>snapchat</th>
      <th>youtube</th>
      <th>facebook</th>
      <th>instagram</th>
      <th>pinterest</th>
      <th>participant</th>
      <th>week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>63</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>66</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



To convert this to long-form, we'll use the "melt" function.

Below is a brief demo of what each week's df will look like, as well as the annotated code.


```python
(
    wk_dfs[0].melt(
        #The values that we want to use as identifiers
        id_vars = ['participant', 'week'],

        #The columns that will unpivot - i.e., form 1 column where each participant will have
        #a row entry for each previous column. Since these are the only columns aside from
        #'participant' and 'week', we technically don't need to even state them, but this is
        #good to just see what's happening.
        value_vars = ['snapchat', 'instagram', 'facebook',
                        'youtube', 'twitter', 'pinterest'],

        #The name of the aforementioned new column - using "platform" to match the same column
        #named on the initial survey (so we can merge later)
        var_name = 'platform',

        #The name of the column for the values; here, the values being hours spent on that
        #platform for that week
        value_name = 'hrs_spent')
    .sort_values('participant')
    .head(12)
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>participant</th>
      <th>week</th>
      <th>platform</th>
      <th>hrs_spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>1</td>
      <td>1</td>
      <td>snapchat</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>314</th>
      <td>1</td>
      <td>1</td>
      <td>facebook</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>719</th>
      <td>1</td>
      <td>1</td>
      <td>pinterest</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>449</th>
      <td>1</td>
      <td>1</td>
      <td>youtube</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>584</th>
      <td>1</td>
      <td>1</td>
      <td>twitter</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>179</th>
      <td>1</td>
      <td>1</td>
      <td>instagram</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>744</th>
      <td>2</td>
      <td>1</td>
      <td>pinterest</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>339</th>
      <td>2</td>
      <td>1</td>
      <td>facebook</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>204</th>
      <td>2</td>
      <td>1</td>
      <td>instagram</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>474</th>
      <td>2</td>
      <td>1</td>
      <td>youtube</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>609</th>
      <td>2</td>
      <td>1</td>
      <td>twitter</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2</td>
      <td>1</td>
      <td>snapchat</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Shortened melt code
for i in range(len(wk_dfs)):
    wk_dfs[i] = wk_dfs[i].melt(id_vars = ['participant', 'week'],
                              var_name = 'platform',
                              value_name = 'hrs_spent').sort_values('participant')
```


```python
#Combining each melted week
weekly_long = pd.concat(wk_dfs).sort_values(['participant', 'week'])
```


```python
#Sanity check: participant 1 should have 18 entries (6 for each week)
weekly_long.head(19)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>participant</th>
      <th>week</th>
      <th>platform</th>
      <th>hrs_spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>1</td>
      <td>1</td>
      <td>twitter</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>314</th>
      <td>1</td>
      <td>1</td>
      <td>youtube</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>719</th>
      <td>1</td>
      <td>1</td>
      <td>pinterest</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>449</th>
      <td>1</td>
      <td>1</td>
      <td>facebook</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>584</th>
      <td>1</td>
      <td>1</td>
      <td>instagram</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>179</th>
      <td>1</td>
      <td>1</td>
      <td>snapchat</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>516</th>
      <td>1</td>
      <td>2</td>
      <td>facebook</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>812</th>
      <td>1</td>
      <td>2</td>
      <td>pinterest</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>1</td>
      <td>2</td>
      <td>twitter</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>664</th>
      <td>1</td>
      <td>2</td>
      <td>instagram</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>368</th>
      <td>1</td>
      <td>2</td>
      <td>youtube</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>220</th>
      <td>1</td>
      <td>2</td>
      <td>snapchat</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>638</th>
      <td>1</td>
      <td>3</td>
      <td>instagram</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>1</td>
      <td>3</td>
      <td>facebook</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>779</th>
      <td>1</td>
      <td>3</td>
      <td>pinterest</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>215</th>
      <td>1</td>
      <td>3</td>
      <td>snapchat</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>356</th>
      <td>1</td>
      <td>3</td>
      <td>youtube</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1</td>
      <td>3</td>
      <td>twitter</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>744</th>
      <td>2</td>
      <td>1</td>
      <td>pinterest</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Things look good! All that's left now is to merge the master weekly df to the initial.

Each participant will still have 6 entries per week, and their initial ratings that they gave each brand will be on that corresponding entry.


```python
master = weekly_long.merge(day_zero_master)

master.head(18)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>participant</th>
      <th>week</th>
      <th>platform</th>
      <th>hrs_spent</th>
      <th>op</th>
      <th>percep</th>
      <th>rec</th>
      <th>use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>twitter</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>twitter</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>twitter</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>youtube</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
      <td>youtube</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>3</td>
      <td>youtube</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>pinterest</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>2</td>
      <td>pinterest</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>3</td>
      <td>pinterest</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>facebook</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>2</td>
      <td>facebook</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>3</td>
      <td>facebook</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1</td>
      <td>instagram</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>2</td>
      <td>instagram</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>3</td>
      <td>instagram</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>1</td>
      <td>snapchat</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>2</td>
      <td>snapchat</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>3</td>
      <td>snapchat</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#master.to_csv('sm_master.csv')
```
