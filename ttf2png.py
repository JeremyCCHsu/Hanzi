# -*- coding: utf-8 -*-
# Note: 
# 	1. Use unichr() and ord() to convert
#   2. *.png is better than *.jpg (which has clear interpolation noise!)
#   3. UTF8 table: http://www.utf8-chartable.de/unicode-utf8-table.pl?start=40704&unicodeinhtml=dec
# 	   2f00 - 2fd5: bu-shou
# 	   4e00 - 9fff: Chinese glyphs (13312 (3400) - 19968 (4E00) - ~40883 - (F6B0) (F900-FADF))
#		3400: most are strange words
# 	   3000 - 30ff: Japanese kana
# 	4. More Chinese fonts: https://magiclen.org/zh-tw-font/ 
#   5. If a glyph is empty, it occupies 109 bytes.

from PIL import Image, ImageDraw, ImageFont

szImg = 28
# iWord = 35069 		# 製 = u'\u88fd' 
oDir = 'TWKai98_%dx%d' % (szImg, szImg)
iFirst = 19968 	# 19968 (Traditional), 13312 (Simplified)
iLast  = 40908
Font   = '../fonts/TW-Kai-98_1.ttf'

font = ImageFont.truetype(Font, size=szImg)
for iWord in range(iFirst, iLast+1):
	print 'Word %d: %s' % (iWord, unichr(iWord))
	im = Image.new("RGB", (szImg, szImg))
	draw = ImageDraw.Draw(im)
	draw.text((0,0), unichr(iWord), font=font)
	im.save('%s/U%d.png' % (oDir, iWord))
