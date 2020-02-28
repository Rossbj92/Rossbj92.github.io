---
title: "Fuzzy String Matching with Pandas and fuzzywuzzy"
excerpt: "Merging datasets with fuzzy matched keys."
tags: [data wrangling, fuzzy matching, fuzzywuzzy, pandas]
header:
  overlay_image: "/images/fuzzy.jpg"
  teaser: "/images/fuzzy.jpg"
read_time: true
comments: true
---
Many times I have had the unfortunate experience of wrangling several datasets with user-entered identifiers (e.g., real names, code names, etc.). In all of these cases, not once have I encountered a project free of errors. Years ago, I would spend hours sifting through spreadsheets, manually linking participants. Even after learning more efficient methods, the scattered misspells would plague my near-perfect merges. The magic of stripping whitespace and lowercasing were no match for the dreaded extra fat-fingered letter.

Something I have found that has saved me hours of eye-strain and new grey hairs is fuzzy matching. In short, fuzzy matching is matching texts that, although not spelled exactly the same, are identical in reality. There are copious ways that this method is used, and the one I use most in my work is matching participant identifiers that have been entered incorrectly.

To illustrate this, let's imagine a simple pre-post study design. Initially, participants complete a survey measuring their overall level of well-being; following the intervention, their well-being is measured again. While their name is entered for them on the pre-experimental survey, the post survey is completed online and requires them to enter their name themselves. This is when the headache begins.

Now, let's generate some data to see what this looks like and what can be done.

*Note: the names below are fictitious and conjured from http://listofrandomnames.com/*


```python
import pandas as pd
```


```python
pre_experiment = pd.DataFrame({'participant':['Marvin Sprouse', 'Shala Pintor', 'Armanda Olivero', 'Shelby Bickett',
                                              'Reinaldo Averitt','Kareem Purser','Oneida Cadogan', 'Davida Pruett',
                                              'Lucas Catanzaro','Ernie Grajeda'],
                               'pre_wellbeing':[4,6,3,5,5,2,7,1,7,10]})
pre_experiment.head()
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
      <th>pre_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Marvin Sprouse</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Shala Pintor</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Armanda Olivero</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Shelby Bickett</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Reinaldo Averitt</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
post_experiment = pd.DataFrame({'participant':['MarvinSprouse', 'Shalla Pintor', 'armanda oliver0', 'shelbey bicket',
                                              'Reinaldo Averitt','Karem Puser','Oneida cadogan', 'Davida Pruettt',
                                              'Lucas Catanzaro','Ernie grajda'],
                               'post_wellbeing':[6,5,5,7,4,9,10,5,7,10]})
post_experiment.head()
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
      <th>post_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>MarvinSprouse</td>
      <td>3</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Shalla Pintor</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>armanda oliver0</td>
      <td>5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>shelbey bicket</td>
      <td>7</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Reinaldo Averitt</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



In this example, we can easily tell which are the correct names, and it also helps that they're in the same order! In actual experimental conditions where there are hundreds or thousands of participants, though, this is not so easy. Pandas has many native string methods that make cleaning text data easier (e.g., ```str.lower()``` can easily handle ```Oneida Cadogan``` and ```Oneida cadogan```) - but instances like ```Ernie Grajeda``` and ```Ernie grajda``` are a bit more difficult.

Again, the end-goal here is to merge our pre and post data to have something we can analyze. To do this, we need the keys (i.e., here those being the ```participant``` column) to be exact matches.

This is what happens if we simply tried to merge the pre and post conditions:


```python
pre_experiment.merge(post_experiment)
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
      <th>pre_data</th>
      <th>post_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Reinaldo Averitt</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Lucas Catanzaro</td>
      <td>9</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>



Only 2 are retained! This is obviously problematic and where fuzzy matching is perfect for the job. We'll be using the fuzzywuzzy library. For installation and more information on the library, you can visit the official github page here (https://github.com/seatgeek/fuzzywuzzy).

Fuzzywuzzy utilizes the Levenshtein Distance to determine string similarity. To put it simply, the Levenshtein Distance is a metric to determine how similar two strings are to eachother based on how many edits are required to transform one into the other.

Let's import what we'll use!


```python
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
```

We'll be using ```process.extract()``` for our matching. First, I'll pass 2 arguments in to demonstrate the output.


```python
process.extract(post_experiment.loc[0,'participant'], pre_experiment['participant'])
```




    [('Marvin Sprouse', 96, 0),
     ('Davida Pruett', 54, 7),
     ('Kareem Purser', 46, 5),
     ('Armanda Olivero', 36, 2),
     ('Shala Pintor', 32, 1)]



One way to read the syntax is that we want to look for a match to ```post_experiment.loc[0,'participant']``` (i.e., ```MarvinSprouse```) in the entire ```participant``` column. We get 5 potential matches in return, with each match containing the actual proposed match, the similarity score, and the corresponding row position of the proposed match.

The potential matches are given in descending order based on similarity, and the first match of ```Marvin Sprouse``` with a similarity score of ```96``` at row ```0``` in ```pre_experiment``` is indeed correct.

We can use ```process.extractOne``` to only return the best match as well. You can also accomplish this using ```limit = 1``` after the last argument (e.g., ```pre_experiment['participant'], limit = 1```).


```python
process.extractOne(post_experiment.loc[0,'participant'], pre_experiment['participant'])
```




    ('Marvin Sprouse', 96, 0)



When I use this method with Pandas dataframes, I like have each part of the output as their own column to spot-check the matches if I suspect any discrepancies. We'll have these automatically generated into columns, but I'll print them out here as an example too.


```python
print(f"Proposed match: {process.extractOne(post_experiment.loc[0,'participant'], pre_experiment['participant'])[0]}")
print(f"Similarity: {process.extractOne(post_experiment.loc[0,'participant'], pre_experiment['participant'])[1]}")
```

    Proposed match: Marvin Sprouse
    Similarity: 96


We can now make a function to apply to the entire dataframe so we don't have to do each individually. A couple of things will happen at each row in ```post_experiment```.
1. That row's ```participant``` will get matched to one in ```pre_experiment``` and placed in the dataframe
2. The similarity score of the match will also be recorded


```python
def fuzzy_match(row):

    row['fuzzy_match'] = process.extractOne(row['participant'], pre_experiment['participant'])[0]
    row['similarity'] = process.extractOne(row['participant'], pre_experiment['participant'])[1]

    return row

```


```python
post_fuzzy = post_experiment.apply(fuzzy_match, axis = 1)
post_fuzzy.head()
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
      <th>post_data</th>
      <th>fuzzy_match</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>MarvinSprouse</td>
      <td>3</td>
      <td>Marvin Sprouse</td>
      <td>96</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Shalla Pintor</td>
      <td>0</td>
      <td>Shala Pintor</td>
      <td>96</td>
    </tr>
    <tr>
      <td>2</td>
      <td>armanda oliver0</td>
      <td>5</td>
      <td>Armanda Olivero</td>
      <td>93</td>
    </tr>
    <tr>
      <td>3</td>
      <td>shelbey bicket</td>
      <td>7</td>
      <td>Shelby Bickett</td>
      <td>93</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Reinaldo Averitt</td>
      <td>4</td>
      <td>Reinaldo Averitt</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



Everything appears to have worked well, and the data are ready to be merged.

Below, we inner join ```pre_experiment``` and ```post_experiment``` based on matching values in ```pre_experiment['participant']``` and ```post_experiment['fuzzy_match']```. For the moment, we need to explicitly state these or Pandas will attempt to use ```post_experiment['participant']```, which will give us the same result as the original merge that we attempted.


```python
pre_post_merge = (
    pre_experiment.merge(post_fuzzy,
                         left_on = 'participant',
                         right_on = 'fuzzy_match')
)

pre_post_merge.head()
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
      <th>participant_x</th>
      <th>pre_data</th>
      <th>participant_y</th>
      <th>post_data</th>
      <th>fuzzy_match</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Marvin Sprouse</td>
      <td>1</td>
      <td>MarvinSprouse</td>
      <td>3</td>
      <td>Marvin Sprouse</td>
      <td>96</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Shala Pintor</td>
      <td>2</td>
      <td>Shalla Pintor</td>
      <td>0</td>
      <td>Shala Pintor</td>
      <td>96</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Armanda Olivero</td>
      <td>3</td>
      <td>armanda oliver0</td>
      <td>5</td>
      <td>Armanda Olivero</td>
      <td>93</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Shelby Bickett</td>
      <td>4</td>
      <td>shelbey bicket</td>
      <td>7</td>
      <td>Shelby Bickett</td>
      <td>93</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Reinaldo Averitt</td>
      <td>5</td>
      <td>Reinaldo Averitt</td>
      <td>4</td>
      <td>Reinaldo Averitt</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



We can do a quick glance as a final sanity check, and everything looks good to go! After just a couple of finishing touches, we'll have a completely clean and merged dataset.


```python
cleaned_merged = (pre_post_merge[['participant_x', 'pre_data', 'post_data']]
                  .rename(columns = {'participant_x':'participant'})
                 )
cleaned_merged.head()
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
      <th>pre_data</th>
      <th>post_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Marvin Sprouse</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Shala Pintor</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Armanda Olivero</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Shelby Bickett</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Reinaldo Averitt</td>
      <td>5</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



I hope that this tutorial has been helpful and that this method saves you as much time as it has saved me. Although fuzzymatching is wonderful, there are still times when entries are too disparate to salvage. Like all techniques, it is also not immune to error.

One method that I use to monitor match failures are left joins and an examination of missing values in the merged data (i.e., the data with fuzzy matches) - this allowing me to see which of the valid names have not appeared in the other data and necessitate further examination.


```python

```
