# Dictionaries represent another way of 
# storing data

# key/value pairs

dict = {"k1": "cat", "k2": "dog", "k3": "mouse", "k4": "fish"}

print(dict['k1'])
print(dict['k3'])
dict['k5'] = "Rabbit"
dict['k2'] = "Mighty Moose"
print(dict)

workers = {'dep_1': 'Peter', 'dep_2': ['Jennifer', 'Michael', 'Tommy']}
print(workers['dep_2'])
print(workers)

Team = {}
Team['PG'] = 'Luka'
Team['SG'] = 'Ant Man'
Team['SF'] = 'Cooper Flagg'
Team['PF'] = 'Kevin Durant'
Team['C'] = 'Wemby'

print(Team['C'])
print(Team.get('SF'))

banking_companies = {"Morgan Stanley": 45.56, "Goldman Sachs": 247.65, "HSBC": 646.35}