# Code- Solve Case Study
I have used 'grade' as X variable and not 'Member_ID'. Only using this (as we have a set across the dataset-A,B,C,D,E etc) helps averaging values and predict the defaulter probability. 'Member_ID'  is not a good Y- parameter because:

1. We cannot consolidate data across large dataset as grouping is not possible, so ML will not learn and always output is 0(default)
2. On trying this method, systems overruns memory as grouping is not possible and too much data affects RAM


Could take sub-grade also for in depth score for better segmentation.
