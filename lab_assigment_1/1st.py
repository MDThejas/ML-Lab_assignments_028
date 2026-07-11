def count_vowel_consonant(text):
    n=len(text)
    v_count=0
    c_count= 0

    for i in range(n):
        if text[i] in ['a','e','i','o','u','A','E','I','O','U']:
            v_count+=1
        else:
            c_count+=1
    return v_count,c_count

text="MDTHEJAS"
vowel_count,consonant_count=count_vowel_consonant(text)
print("vowel count is ",vowel_count)
print("consonant count is ",consonant_count)