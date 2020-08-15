def generate_sentences(data):
    '''
    input dataframe.
      user  item
    	1	a
    	1	b
    	1	c
    	2	d
    	2	a
    output sentences. [user_id1, all_item(1...k1),user_id2,all_item(1...k2),...]
    [1,a,b,c,2,d,a]
    '''
    out = []
    data['item'] = data['item'].astype(str)
    data['user'] = data['user'].astype(str)
    for user in data.user.unique():
        temp = [user]
        temp.extend(data[data.user == user].item.unique().tolist())
        out.append(temp)
    return out
