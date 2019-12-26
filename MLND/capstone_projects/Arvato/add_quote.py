with open("artributes_in_excel.csv") as i: # open file for reading, i = input file 
  with open("artributes_in_excel_2.csv", "w") as o: # open temp file in write mode, o = output 
     for l in i: # read line by line
         o.write("'%s',\n" % l[:-1]) # concate ' and text 
          #       ^  ^ added `'` for each line  
# os.remove("file") # delete old file. Note:this is not needed in postfix system 
# os.rename("temp", "file")  # rename file