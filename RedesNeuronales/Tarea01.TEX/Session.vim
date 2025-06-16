let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Documents/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/RedesNeuronales/Tarea01.TEX
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +174 MarquezSalazarBrandon-RedesNeuronales-Tarea01.tex
badd +52 bibliography.bib
badd +33 Macros.tex
badd +0 notebook/perceptron.qmd
argglobal
%argdel
tabnew +setlocal\ bufhidden=wipe
tabrewind
edit MarquezSalazarBrandon-RedesNeuronales-Tarea01.tex
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
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
exe 'vert 1resize ' . ((&columns * 130 + 134) / 268)
exe 'vert 2resize ' . ((&columns * 137 + 134) / 268)
argglobal
setlocal foldmethod=expr
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
27
sil! normal! zo
41
sil! normal! zo
89
sil! normal! zo
96
sil! normal! zo
104
sil! normal! zo
109
sil! normal! zo
121
sil! normal! zo
152
sil! normal! zo
169
sil! normal! zo
let s:l = 52 - ((31 * winheight(0) + 34) / 69)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 52
normal! 017|
wincmd w
argglobal
if bufexists(fnamemodify("Macros.tex", ":p")) | buffer Macros.tex | else | edit Macros.tex | endif
if &buftype ==# 'terminal'
  silent file Macros.tex
endif
balt bibliography.bib
setlocal foldmethod=expr
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
let s:l = 33 - ((32 * winheight(0) + 34) / 69)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 33
normal! 09|
wincmd w
exe 'vert 1resize ' . ((&columns * 130 + 134) / 268)
exe 'vert 2resize ' . ((&columns * 137 + 134) / 268)
tabnext
edit notebook/perceptron.qmd
argglobal
balt MarquezSalazarBrandon-RedesNeuronales-Tarea01.tex
setlocal foldmethod=syntax
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
let s:l = 1 - ((0 * winheight(0) + 34) / 69)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1
normal! 0
tabnext 2
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
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
