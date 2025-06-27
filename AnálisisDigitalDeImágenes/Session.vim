let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes/Tarea01.TEX
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +4 code/src/main.cpp
badd +1 code/dibujo.cxx
badd +1 code/dibujo.hxx
badd +742 code/src/OCV_Course.cxx
badd +30 code/src/dibujo.cxx
badd +9 code/src/dibujo.hxx
argglobal
%argdel
edit code/src/dibujo.hxx
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 30 + 134) / 268)
exe '2resize ' . ((&lines * 38 + 40) / 80)
exe 'vert 2resize ' . ((&columns * 118 + 134) / 268)
exe '3resize ' . ((&lines * 38 + 40) / 80)
exe 'vert 3resize ' . ((&columns * 118 + 134) / 268)
exe 'vert 4resize ' . ((&columns * 118 + 134) / 268)
argglobal
enew
file NvimTree_1
balt code/src/main.cpp
setlocal foldmethod=syntax
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal nofoldenable
wincmd w
argglobal
balt code/src/dibujo.cxx
setlocal foldmethod=expr
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
1
sil! normal! zo
let s:l = 10 - ((9 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 10
normal! 0
lcd ~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes
wincmd w
argglobal
if bufexists(fnamemodify("~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes/Tarea01.TEX/code/src/dibujo.cxx", ":p")) | buffer ~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes/Tarea01.TEX/code/src/dibujo.cxx | else | edit ~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes/Tarea01.TEX/code/src/dibujo.cxx | endif
if &buftype ==# 'terminal'
  silent file ~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes/Tarea01.TEX/code/src/dibujo.cxx
endif
balt ~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes/Tarea01.TEX/code/src/dibujo.hxx
setlocal foldmethod=expr
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
3
sil! normal! zo
let s:l = 24 - ((23 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 24
normal! 06|
lcd ~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes
wincmd w
argglobal
if bufexists(fnamemodify("~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes/Tarea01.TEX/code/src/main.cpp", ":p")) | buffer ~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes/Tarea01.TEX/code/src/main.cpp | else | edit ~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes/Tarea01.TEX/code/src/main.cpp | endif
if &buftype ==# 'terminal'
  silent file ~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes/Tarea01.TEX/code/src/main.cpp
endif
balt ~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/AnálisisDigitalDeImágenes/Tarea01.TEX/code/src/dibujo.hxx
setlocal foldmethod=expr
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
8
sil! normal! zo
let s:l = 10 - ((9 * winheight(0) + 38) / 77)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 10
normal! 0
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 30 + 134) / 268)
exe '2resize ' . ((&lines * 38 + 40) / 80)
exe 'vert 2resize ' . ((&columns * 118 + 134) / 268)
exe '3resize ' . ((&lines * 38 + 40) / 80)
exe 'vert 3resize ' . ((&columns * 118 + 134) / 268)
exe 'vert 4resize ' . ((&columns * 118 + 134) / 268)
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
nohlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
