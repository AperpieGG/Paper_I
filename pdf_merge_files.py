# import PyPDF2

# # Open the two PDF files you want to combine
# pdf1 = open('/Users/u5500483/Documents/GitHub/cmos/plotting_tools/Dark_Current_FFR_12bit_temp_-25.pdf', 'rb')
# pdf2 = open('/Users/u5500483/Documents/GitHub/cmos/plotting_tools/Dark_Current_HDR_12bit_temp_-25.pdf', 'rb')
#
# pdf_reader1 = PyPDF2.PdfFileReader(pdf1)
# pdf_reader2 = PyPDF2.PdfFileReader(pdf2)

# # Create a new PDF to write the output
# pdf_writer = PyPDF2.PdfFileWriter()

# # Get the first page from each PDF
# page1 = pdf_reader1.getPage(0)
# page2 = pdf_reader2.getPage(0)

# # Get the page size (optional)
# width = page1.mediaBox.getWidth() + page2.mediaBox.getWidth()
# height = max(page1.mediaBox.getHeight(), page2.mediaBox.getHeight())

# # Create a new page and add the first two pages side by side
# output_page = pdf_writer.addBlankPage(width, height)
# output_page.mergeTranslatedPage(page1, 0, 0)
# output_page.mergeTranslatedPage(page2, page1.mediaBox.getWidth(), 0)

# # Write the new PDF to a file
# output_pdf = open('plotting_tools/DC_FFR_HDR.pdf', 'wb')
# pdf_writer.write(output_pdf)

# # Close the PDF files
# pdf1.close()
# pdf2.close()
# output_pdf.close()


import PyPDF2

# Open the two PDF files you want to combine
pdf1 = open('/Results/DC_vs_Temp.pdf', 'rb')
pdf2 = open('/Results/Dark_FFR_T.pdf', 'rb')

pdf_reader1 = PyPDF2.PdfReader(pdf1)
pdf_reader2 = PyPDF2.PdfReader(pdf2)

# Create a new PDF to write the output
pdf_writer = PyPDF2.PdfWriter()

# Get the first page from each PDF
page1 = pdf_reader1.pages[0]
page2 = pdf_reader2.pages[0]

# Get the page size (optional)
width = max(page1.mediabox.getWidth(), page2.mediabox.getWidth())
height = page1.mediabox.getHeight() + page2.mediabox.getHeight()

# Create a new page and add the first two pages, one on top of the other
output_page = pdf_writer.addBlankPage(width, height)
output_page.mergeTranslatedPage(page1, 0, 0)  # Position page1 at the top
output_page.mergeTranslatedPage(page2, 0, page1.mediabox.getHeight())  # Position page2 below page1

# Write the new PDF to a file
output_pdf = open('Results/Dark_FFR_EXP_T.pdf', 'wb')
pdf_writer.write(output_pdf)

# Close the PDF files
pdf1.close()
pdf2.close()
output_pdf.close()




