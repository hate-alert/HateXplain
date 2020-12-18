import re 
from Preprocess.preProcess import *
from Preprocess.preProcess import preProcessing

from transformers import BertTokenizer
import string 
#input:Given a string in the form of "(start1-end1)span_text1||(start2-end2)......"  or "{}" if no span selected 
#output:Gives back a list of tuples of the form (string,start,end) or [] if no span selected

# def giveSpanList(string1):
#     if string1 in ["{}","{","}"]  :
#         return []
#     list1=string1.split("||")
#     string_all=[]
#     for l in list1:
#     	# collect the string 
#         string=re.sub(r'\([^)]*\)', '', l)
#         # colect the string postion (start--end) in the original text
#         string_mask=re.findall('\((.*?)\)',l)[0]
#         [start,end]=string_mask.split("--")
#         string_all.append((string,start,end))
#     return string_all 

def giveSpanList(row,tokens,string1,data_type):
    if(data_type=='old'):
        if string1 in ["{}","{","}"]  :
            return []
        list1=string1.split("||")
        string_all=[]
        for l in list1:
            # collect the string 
            # colect the string postion (start--end) in the original text
            string_mask=re.findall('\((.*?)\)',l)[0]
            string=l[len(string_mask)+2:]
            [start,end]=string_mask.split("--")
            string_all.append((string,start,end))
    else:
        string_all=[]
        flag=0
        if(row['post_id'] in ['10510109_gab','1081573659137830912_twitter','1119979940080783360_twitter']):
              flag=1
        for exp in string1:
            start,end= int(exp.split('-')[1]),int(exp.split('-')[2])
            if(flag==1):
                print(exp)

            start_pos=0
            end_pos=0
            pos=0
            count=0
            for tok in tokens:
                if(flag==1):
                    print(count)
                    print(pos)
                if(count==start):
                    start_pos=pos
                pos+=len(tok)+1
                if((count+1)==end):
                    end_pos=pos
                    break
                
                count+=1
                
            string_all.append((exp.split('-')[0],start_pos,end_pos)) 
        
    return string_all






def returnMask(row,params,tokenizer):
    
    text_tokens=row['text']
    
    
    
    ##### a very rare corner case
    if(len(text_tokens)==0):
        text_tokens=['dummy']
        print("length of text ==0")
    #####
    
    
    mask_all= row['rationales']
    mask_all_temp=mask_all
    count_temp=0
    while(len(mask_all_temp)!=3):
        mask_all_temp.append([0]*len(text_tokens))
    
    word_mask_all=[]
    word_tokens_all=[]
    
    for mask in mask_all_temp:
        if(mask[0]==-1):
            mask=[0]*len(mask)
        
        
        list_pos=[]
        mask_pos=[]
        
        flag=0
        for i in range(0,len(mask)):
            if(i==0 and mask[i]==0):
                list_pos.append(0)
                mask_pos.append(0)
            
            
            
            
            if(flag==0 and mask[i]==1):
                mask_pos.append(1)
                list_pos.append(i)
                flag=1
                
            elif(flag==1 and mask[i]==0):
                flag=0
                mask_pos.append(0)
                list_pos.append(i)
        if(list_pos[-1]!=len(mask)):
            list_pos.append(len(mask))
            mask_pos.append(0)
        string_parts=[]
        for i in range(len(list_pos)-1):
            string_parts.append(text_tokens[list_pos[i]:list_pos[i+1]])
        
        
        
        
        if(params['bert_tokens']):
            word_tokens=[101]
            word_mask=[0]
        else:
            word_tokens=[]
            word_mask=[]

        
        for i in range(0,len(string_parts)):
            tokens=ek_extra_preprocess(" ".join(string_parts[i]),params,tokenizer)
            masks=[mask_pos[i]]*len(tokens)
            word_tokens+=tokens
            word_mask+=masks


        if(params['bert_tokens']):
            ### always post truncation
            word_tokens=word_tokens[0:(int(params['max_length'])-2)]
            word_mask=word_mask[0:(int(params['max_length'])-2)]
            word_tokens.append(102)
            word_mask.append(0)

        word_mask_all.append(word_mask)
        word_tokens_all.append(word_tokens)
        
#     for k in range(0,len(mask_all)):
#          if(mask_all[k][0]==-1):
#             word_mask_all[k] = [-1]*len(word_mask_all[k])
    if(len(mask_all)==0):
        word_mask_all=[]
    else:    
        word_mask_all=word_mask_all[0:len(mask_all)]
    return word_tokens_all[0],word_mask_all    
        
        
        
        
               
            
            


















#### NEED ONLY WHEN RAW INPUTS FROM ANNOTATORS.
#INPUTS: text,mask_all,debug 
#text contains the text in the dataset
#mask all contains the list of attention span from all the annotaters
#debug is set to true if print results are required  
#OUTPUTS: outputs word /sub word tokens tokens , 
def returnMaskonetime(row,params,tokenizer,debug=False,data_type='old'):
    text = row['text']
    mask_all= [row['explain1'],row['explain2'],row['explain3']]
    data_type = row['old_vs_new']
    if(data_type == 'old'):
        ### convert mentions to @user
        text = re.sub('@\w+', '@user',text)
        #### remove any html spans if present in the text    	
        text = cleanhtml(text)
        ###remove some errorneous characters in the text
        text = re.sub(u"(\u2018|\u2019|\u201A|\u201B|\u201C|\u201D|\u201E)", "'", text)
        text = text.replace("\r\n",' ').replace("\n",' ')
        #text = text.replace(u'\xa0', u' ')
        ####initialize word mask and word tokens
        list_text=text.split(" ")
    else:
        line = ' '.join(preProcessing(str(text).strip()))
        line = line.replace("\"", "\"\"")
        #text = line
        list_text=line.split()
        
    word_mask_all=[]
    word_tokens_all=[]
    string_parts_all=[]
    list_pos_all=[]
    span_list_all=[]
    mask_pos_all=[]
    for mask in mask_all:
        if(row['final_annotation']=='normal'):
            string_all=[]
        else:
            string_all=giveSpanList(row,list_text,mask,data_type)
        
        if(row['old_vs_new']=='old'):
            spanlist= []
            for ele in string_all:
                if(ele==[]):
                    continue
                #print(ele[0],ele[1],ele[2])
                text_temp=" ".join(list_text)
                #print(text_temp)
                try:
                    res = [i.start() for i in re.finditer(ele[0],text_temp)]
                except:
                    res =[]
                if(len(res)==0):
                    #### for bad characters in docanno
                    ele_temp = ele[0].replace('’', '\'')
                    ele_temp = ele_temp.replace('”', '\'')
                    try:
                        res = [i.start() for i in re.finditer(ele_temp,text_temp)]
                    except:
                        res =[]

                if(len(res)==0):
                    ### for spurious \n cases
                    ele_temp = ele[0].replace('\n', '')
                    try:
                        res = [i.start() for i in re.finditer(ele_temp,text_temp)]
                    except:
                        res =[]

                if(len(res)==0):
                    ### for spurious \n cases
                    ele_temp = ele[0].replace('\n', ' ')
                    try:
                        res = [i.start() for i in re.finditer(ele_temp,text_temp)]
                    except:
                        res =[]

                if(len(res)==0):
                    #escape characters in the text.
                    res = [i.start() for i in re.finditer(re.escape(ele[0]),text_temp)]
                if(len(res)==0):
                    #Try direct matching also
                    if(text_temp[int(ele[1]):int(ele[2])]==ele_temp):
                        res=[ele[1]]
                if(len(res)==0):
                    print(row['post_id'])
                    #spanlist.append(ele)
                    #continue
                    #list_errors.append(row['post_id'])
                    #return "unsolved"
                flag=0
                current_lowest = 1000
                current_pos= 0
                for res_ele in res:  
                    diff=abs(int(ele[1])-res_ele)
                    if(diff<=current_lowest):
                        current_pos=res_ele
                        current_lowest=diff
                        
                end_pos=current_pos+len(ele[0].rstrip())

                #### making partial annotation to full annotations
                while True:
                    if (current_pos <= 0):
                        current_pos=0
                        break

                    if (text_temp[current_pos-1]==" "):
                        break
                    if(text_temp[current_pos-1]!=" "):
                        current_pos=current_pos-1

                while True:

                    if (end_pos >= len(text_temp)):
                        end_pos=len(text_temp) 
                        break

                    if (text_temp[end_pos]==" "):
                        end_pos=end_pos+1
                        break
                    if(text_temp[end_pos]!=" "):
                        end_pos=end_pos+1
                spanlist.append([ele[0],current_pos,end_pos]) 
        else:
            text=' '.join(list_text)
            spanlist=string_all
        
        total_length=len(text)
        
        #print(spanlist)
        
        spanlist.sort(key = lambda elem: int(elem[1]))
        
        
        
        list_pos=[]
        mask_pos=[]
        
        for i in range(0,len(spanlist)):
            ele=spanlist[i]
            if(i<len(spanlist)-1):
                ele_next = spanlist[i+1]
            else:
                ele_next = ['',total_length,total_length]
            if(i==0):
                if(ele[1]!= 0):
                    list_pos.append(0)
                    list_pos.append(int(ele[1]))
                    mask_pos.append(0)
                    mask_pos.append(1)
                else:
                    list_pos.append(int(ele[1]))
                    mask_pos.append(1)
            else:
                list_pos.append(int(ele[1]))
                mask_pos.append(1)
            
            if(int(ele_next[1]) > int(ele[2])):
                list_pos.append(int(ele[2]))
                mask_pos.append(0)
            
            
        if(len(list_pos)==0):
            list_pos.append(0)
            mask_pos.append(0)
            list_pos.append(total_length)
            mask_pos.append(0)
        elif(list_pos[-1]!=total_length):
            list_pos.append(total_length)
            mask_pos.append(0)
#         list_pos_modified=[list_pos[0]]
#         for pos in list_pos[1:]:
#             if list_pos_modified[-1] != pos:
#                    list_pos_modified.append(pos)
                    
#         list_pos=list_pos_modified 
        
        
        
        string_parts=[]
        for i in range(len(list_pos)-1):
            string_parts.append(text[list_pos[i]:list_pos[i+1]])
        
        
        string_parts_all.append(string_parts)
        list_pos_all.append(list_pos)
        span_list_all.append(spanlist)
        mask_pos_all.append(mask_pos)
        if(params['bert_tokens']):
            word_tokens=[101]
            word_mask=[0]
        else:
            word_tokens=[]
            word_mask=[]

        if(debug==True):     
            print(list_pos)
            print(string_parts)

        for i in range(0,len(string_parts)):
            tokens=ek_extra_preprocess(string_parts[i],params,tokenizer)
            masks=[mask_pos[i]]*len(tokens)
            word_tokens+=tokens
            word_mask+=masks


        if(params['bert_tokens']):
            ### always post truncation
            word_tokens=word_tokens[0:(int(params['max_length'])-2)]
            word_mask=word_mask[0:(int(params['max_length'])-2)]
            word_tokens.append(102)
            word_mask.append(0)

        word_mask_all.append(word_mask)
        word_tokens_all.append(word_tokens)
    if(row['old_vs_new']=='new'):
        if(row['final_annotation'] in ['normal','non-toxic']):
            for i in range(0,3):
                word_mask_all[i]=[-1]*len(word_mask_all[i])
        else:
            word_mask_all[2] = [-1]*len(word_mask_all[2])
    else:
        for i in range(0,3):
            if(row['pred'+str(i+1)] in ['normal','non-toxic']):
                word_mask_all[i]=[-1]*len(word_mask_all[i])
        
    #print(len(word_tokens_all))    
    return word_tokens_all[0],word_mask_all,string_parts_all,list_pos_all,span_list_all,mask_pos_all
   





