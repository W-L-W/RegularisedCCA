# first download data from https://tibshirani.su.domains/PMA/
# as instructed in https://cran.r-project.org/web/packages/PMA/PMA.pdf
load("~/Downloads/breastdata.rda")
fn <- function(x) {
  x + 1 # A comment, kept as part of the source
}	
setwd(utils::getSrcDirectory(fn)[1])

# # first attempt - put everything together:
# dna_temp <- cbind(breastdata$chrom,breastdata$nuc,breastdata$dna)
# 
# patients <- sapply(1:89,function(n){paste0('p',n)})
# colnames(dna_temp) <- c('chrom','nuc',patients)
# 
# dna_temp[1:10,1:10]
# 
# rna_temp <- cbind(breastdata$gene,breastdata$genenames,breastdata$genechr,
#               breastdata$genepos,breastdata$genedesc,breastdata$rna)
# colnames(rna_temp) <- c('gene','genename','genechr','genepos','genedesc',patients)
# rna_temp[1:3,1:7]

# I prefer second idea - separate data from label information:
dna_matrix <- breastdata$dna
dna_labels <- cbind(breastdata$chrom,breastdata$nuc)
colnames(dna_labels) <- c('chrom','nuc')
write.csv(dna_labels, file='dna_labels.csv')
