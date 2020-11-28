import matplotlib.pyplot as plt
from matplotlib import transforms


def normalize_attentions(attentions, starting_offsets, ending_offsets):
    cum_attn = attentions.cumsum(1)
    start_v = cum_attn.gather(1, starting_offsets - 1)
    end_v = cum_attn.gather(1, ending_offsets - 1)
    return end_v - start_v


def get_start_idxs_batched(b_tokens, b_sub_tokens, bpe_indicator='Ġ'):
    
    # print(b_sub_tokens[0])

    from copy import deepcopy as cp
    ss = []
    es = []
    batch_size = len(b_tokens)
    for i in range(batch_size):
        starts = [1]
        starts += list(filter(lambda j: b_sub_tokens[i][j][0] == bpe_indicator, range(len(b_sub_tokens[i]))))
        assert len(starts) == len(b_tokens[i])
        ends = cp(starts)
        ends.append(len(b_sub_tokens[i]) - 1)
        ends = ends[1:]
        # starts = [0] + starts + [len(b_sub_tokens[i]) - 1]
        # ends = [1] + ends + [len(b_sub_tokens[i])]
        ends = list(filter(lambda x: x <= 512, ends))       # less than 512
        ss.append(starts[:len(ends)])
        es.append(ends)
    return ss, es


def rainbow_text(x, y, strings, colors, bgcolors, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas
    for s, c, bgc in zip(strings, colors, bgcolors):
        text = ax.text(x, y, s+" ", color=c, backgroundcolor=bgc,
                       transform=t, **kwargs)
        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(
            text.get_transform(), x=ex.width, units='dots')


def gen_score_img(raw, score, label, path=None):
    
    plt.figure()
    plt.axis('off')
    size = 12
    ntab = 0
    nline = 1
    sep = 0.105
    xinit = 0.02
    yinit = 0.98
    rainbow_text(xinit, yinit, ["Class %d" % (label)],
                    ["black"], ["white"], size=size)
    words, colors, bgcolors = [], [], []
    nonewline, nbrace, flush = False, 0, False
    assert len(raw) == len(score)
    for t, h, i in zip(raw, score, range(len(score))):
        if flush:
            flush = False
            if t != "{":
                rainbow_text(xinit+ntab*sep, yinit-nline*sep, words,
                             colors, bgcolors, size=size)
                nline += 1
                words, colors, bgcolors = [], [], []
        words.append(t)
        bgc = 1-float(h)
        bgcolors.append((bgc, bgc, bgc))
        if bgc > 0.5:
            colors.append((0, 0, 0))
        else:
            colors.append((1, 1, 1))
        if t == '}':
            ntab -= 1
            rainbow_text(xinit+ntab*sep, yinit-nline*sep, words,
                         colors, bgcolors, size=size)
            nline += 1
            words, colors, bgcolors = [], [], []
        elif t == "{":
            rainbow_text(xinit+ntab*sep, yinit-nline*sep, words,
                         colors, bgcolors, size=size)
            ntab += 1
            nline += 1
            words, colors, bgcolors = [], [], []
        elif t == ";" and (not nonewline):
            rainbow_text(xinit+ntab*sep, yinit-nline*sep, words,
                         colors,  bgcolors, size=size)
            nline += 1
            words, colors, bgcolors = [], [], []
        elif t in ["for", "while", "if"]:
            nonewline = True
            nbrace = 0
        elif t == "(" and nonewline:
            nbrace += 1
        elif t == ")" and nonewline:
            nbrace -= 1
            if nbrace <= 0:
                flush = True
                nonewline = False
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
    plt.close()