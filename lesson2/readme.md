Computer vision - deeper appplication

Ideas discussed in first week:
1. Turning sound into picture and then applying fast ai

2. Turing lab results into picture and applying fast ai

3. whatsapp images discrimination based on spamand not spam

4. Using it to classify dog breeds as hairy and non hairy

5. using it to identify building that are complete or not complete

6. TO identify if a partiular place belongs to india
trhough sattelite image

If youre stuck KEep going.

This javascript code can be used to download all links and put it in a text file

urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));


### Errata

I found that ImageDataBunch wasonly picking up one folder when the actual required behavior was to pich from all subfolders, It is picking however the labels are all the same folder name

At 25 minutes
https://course.fast.ai/videos/?lesson=2

