import matplotlib.pyplot as plt
import numpy as np
import os
from plotter import Plotter
from pystog import Converter
from pystog import FourierFilter
from pystog import Pre_Proc
from pystog import Transformer


def signaltonoise(a, axis=0, ddof=0):
    """Calculate the signal-to-noise (SNR) ratio of an array.

    The signal-to-noise ratio is defined as the mean of the array
    divided by its standard deviation.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which the SNR is computed. The default is 0.
        If None, compute the SNR of the flattened array.
    ddof : int, optional
        Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        Default is 0.

    Returns
    -------
    ndarray
        The signal-to-noise ratio of `a`. If the standard deviation is 0,
        the SNR is returned as 0.
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)

    return np.where(sd == 0, 0, m/sd)


def is_equally_spaced(arr, tolerance=1e-9):
    """
    Checks if a 1D NumPy array is equally spaced.

    Parameters
    ----------
    arr : numpy.ndarray
        The input 1D array.
    tolerance : float, optional
        The maximum allowed difference between spacing intervals
        for them to be considered equal. Default is 1e-9.

    Returns
    -------
    float
        The spacing between elements if the array is equally spaced,
        otherwise -1.0.
    """
    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional.")
    if arr.size < 2:
        return True  # Arrays with 0 or 1 element are considered equally spaced

    diffs = np.diff(arr)

    if np.allclose(diffs, diffs[0], atol=tolerance):
        return arr[1] - arr[0]
    else:
        return -1.


def validate_input(data):
    """Validate the input data for running the chunk-by-chunk algorithm.

    Parameters
    ----------
    data : dict
        The input data to be validated.

    Returns
    -------
    bool
        True if the input data is valid, False otherwise.
    """
    files = data.get('Files', None)
    if files is None:
        print("[Error] Input data must contain 'Files' key.")
        return False
    num_density = data.get('NumberDensity', None)
    if num_density is None:
        print("[Error] Input data must contain 'NumberDensity' key.")
        return False
    output_stem = data.get('OutputStem', None)
    if output_stem is None:
        print("[Error] Input data must contain 'OutputStem' key.")
        return False
    q_out_form = data.get('QSpaceOutputForm', "S(Q)")
    r_out_form = data.get('RSpaceOutputForm', "g(r)")
    check_fz = False
    if "K" in q_out_form or "K" in r_out_form:
        check_fz = True
        if "FaberZiman" in data:
            fzcoeff = data["FaberZiman"]
        else:
            print(
                "[Error] FaberZiman must be provided for "
                "output in Keen's forms."
            )
            return False

    condt1 = not isinstance(files, list)
    condt2 = not all(isinstance(f, str) for f in files)
    if condt1 or condt2:
        print("[Error] Files must be a list of strings.")
        return False

    # If number density is given as a single value, it will apply to all files.
    if not isinstance(num_density, (int, float, list)):
        print("[Error] NumberDensity must be an int, float, or list.")
        return False
    else:
        if not isinstance(num_density, list):
            num_density = [num_density] * len(files)

    # If Faber-Ziman coefficient is given as a single value, it will apply to
    # all files.
    if check_fz:
        if not isinstance(fzcoeff, (int, float, list)):
            print("[Error] NumberDensity must be an int, float, or list.")
            return False
        else:
            if not isinstance(fzcoeff, list):
                fzcoeff = [fzcoeff] * len(files)

    condt1 = not isinstance(output_stem, list)
    condt2 = not all(isinstance(f, str) for f in output_stem)
    if condt1 or condt2:
        print("[Error] OutputStem must be a list of strings.")
        return False

    if len(files) > 1:
        if len(files) != len(output_stem) and len(output_stem) != 1:
            print(
                "[Error] If multiple files are provided, OutputStem must "
                "match the number of files."
            )
            return False
        if len(files) != len(num_density):
            print(
                "[Error] If multiple files are provided, NumberDensity must "
                "match the number of files."
            )
            return False
        if check_fz and len(files) != len(fzcoeff):
            print(
                "[Error] If multiple files are provided, FaberZiman must "
                "match the number of files."
            )
            return False

    if "QChunks" not in data:
        print("[Error] Input data must contain 'QChunks' key.")
        return False

    if "RChunks" not in data:
        print("[Error] Input data must contain 'RChunks' key.")
        return False

    qchunks = data["QChunks"]
    rchunks = data["RChunks"]
    if not isinstance(qchunks, list):
        print("[Error] QChunks must be a list of numbers.")
        return False
    if not isinstance(rchunks, list):
        print("[Error] RChunks must be a list of numbers.")
        return False
    if len(qchunks) != len(rchunks):
        print(
            "[Error] QChunks and RChunks must have the same length."
        )
        return False

    return True


def rebin_data(x, y, bin_size=0.01):
    """Rebin the data to a specified bin size.

    Parameters
    ----------
    x : numpy.ndarray
        The x-coordinates of the data.
    y : numpy.ndarray
        The y-coordinates of the data.
    bin_size : float
        The size of the bins to rebin the data to.

    Returns
    -------
    tuple
        The rebinned x and y data.
    """
    bin_in = is_equally_spaced(x)

    # Return of '-1' means the data is not equally spaced.
    if bin_in == -1. or bin_in != bin_size:
        rebin = Pre_Proc.rebin
        x_min = x.min()
        x_max = x.max()
        x_rebin, y_rebin = rebin(x, y, x_min, bin_size, x_max)

        if isinstance(x_rebin, list):
            x_rebin = np.asarray(x_rebin)
        if isinstance(y_rebin, list):
            y_rebin = np.asarray(y_rebin)

        return (x_rebin, y_rebin)
    else:
        return (x, y)


def run_stog_ck(
        file, num_density, output, hlines,
        input_form, qmin, qbin, qchunks,
        rbin, rchunks, interactive, diagnostic,
        rs_min=None, rs_max=None, r_cut=None, fzcoeff=None,
        rmax_out=50., q_out_form='S(Q)', r_out_form='g(r)'):
    """Run the chunk-by-chunk Fourier transform processing.

    Parameters
    ----------
    file : str
        The input file name.
    num_density : float
        The number density for the calculation.
    output : str
        The output file name.
    hlines : int
        The number of header lines in the input data file.
    input_form : str
        The input form of the data. Can be 'S(Q)-1' or 'S(Q)'.
    qmin : float
        The minimum q value for Fourier transform.
    qbin : float
        The bin size for the q-axis.
    qchunks : list
        The list of q-chunks for the calculation.
    rbin : float
        The bin size for the r-axis.
    rchunks : list
        The list of r-chunks for the calculation.
    interactive : bool
        Whether to run in interactive mode or not.
    diagnostic : bool
        Whether to run in diagnostic mode or not.
    rs_min : float, optional
        The minimum r value for scaling. Default is None.
    rs_max : float, optional
        The maximum r value for scaling. Default is None.
    r_cut : float, optional
        The cutoff value for the Fourier filter. Default is None.
    fzcoeff : float, optional
        The Faber-Ziman coefficient. Default is None.
    rmax_out : float, optional
        The maximum r value for the output. Default is 50.
    q_out_form : str, optional
        The output form for the q-space data. Default is 'S(Q)'.
    r_out_form : str, optional
        The output form for the r-space data. Default is 'g(r)'.

    Returns
    -------
    None
        The function does not return anything. It writes to output files.
    """
    q, sq = np.loadtxt(
        file, skiprows=hlines, unpack=True
    )
    (q, sq) = rebin_data(q, sq, qbin)
    if input_form == "S(Q)-1":
        sq += 1.

    q_init = np.copy(q)
    sq_init = np.copy(sq)
    qmax_init = q.max()

    plot_handle = Plotter()
    transformer = Transformer()
    ff = FourierFilter()
    converter = Converter()
    color_list = [
        "black", "red", "blue",
        "green", "orange", "purple",
        "brown", "pink", "gray",
        "cyan", "magenta", "yellow"
    ]

    # The rmax value for the inverse Fourier transform, according to the
    # Nyquist-Shannon theorem. See the post below for a bit explanation
    # about the factor of 2,
    # https://iris2020.net/2020-06-28-fourier_transform
    rmax = np.pi / qbin
    rmax_idx = int(rmax / rbin)

    r_raw = np.linspace(0., rmax, rmax_idx)
    q2, sq2, _ = transformer.apply_cropping(q, sq, qmin, qchunks[0])
    # This is for the offset of S(Q) by the average of the high Q region then
    # plus 1 to make sure the high Q region is oscillating around 1.
    sq2 = sq2 - np.mean(sq2[len(sq2) - int(5. / qbin): len(sq2)]) + 1.
    r, gr, _ = transformer.S_to_g(q2, sq2, r_raw, rho=num_density)
    gr -= 1.

    # Check and decide the low r region for scaling, interactively.
    if rs_min is None or rs_max is None or interactive:
        plot_handle.update_data(
            [r],
            [gr],
            ["Initial Fourier Transform"],
            ["black"],
            "r",
            "g(r) - 1",
            "Å",
            "",
            title="Low-r region inspection for scaling"
        )
        _, ax = plot_handle.plot()

        ax.set_xlim(0, 5)
        ax.set_ylim(-3, 3)

        plt.show(block=False)

        rs_range = input("\n[Input] r range for scaling (e.g., 1.1, 1.5): ")
        if "," in rs_range:
            rs_min, rs_max = map(float, rs_range.split(","))
        else:
            rs_min, rs_max = map(float, rs_range.split())

        plt.close()

    # The idea for scaling is based on the fact that the low-r region of
    # g(r) - 1 should be oscillating around 1. Therefore, the average value
    # of the low-r region within the specified boundary can be used as scale
    # factor.
    scale = -np.mean(gr[int(rs_min / rbin): int(rs_max / rbin)])
    sq /= scale

    # Initial Fourier transform of the scaled data.
    q2, sq2, _ = transformer.apply_cropping(q, sq, qmin, qchunks[0])
    sq2 = sq2 - np.mean(sq2[len(sq2) - int(5. / qbin): len(sq2)]) + 1.
    r, gr, _ = transformer.S_to_g(q2, sq2, r_raw, rho=num_density)

    # To interactively decide the cutoff for the Fourier filter. Alternative
    # positions are those intersections of the g(r) with the x-axis.
    if r_cut is None or interactive:
        sign_crossing = np.where(np.diff(np.signbit(gr)))[0]
        plot_handle.update_data(
            [r],
            [gr - 1.],
            ["Initial Fourier Transform"],
            ["black"],
            "r",
            "g(r) - 1",
            "Å",
            "",
            title="Low-r region inspection for Fourier filtering"
        )
        fig, ax = plot_handle.plot()

        ax.scatter(
            r[sign_crossing],
            -1. + np.zeros(len(sign_crossing)),
            color='blue'
        )
        ax.axhline(y=-1, color="red")
        ax.axvline(x=rs_min, color="green", linestyle="--")
        ax.axvline(x=rs_max, color="green", linestyle="--")
        ax.annotate(
            "r-range used for scaling",
            xy=((rs_min + rs_max) / 2.0, 3.0),
            ha="center",
            fontsize=18,
            color="green",
        )
        ax.set_xlim(0., r[sign_crossing[-1]] + 1.)
        ax.set_ylim(-4., 4.)
        plt.show(block=False)

        print("\n============================================================")
        print("Data scaled based on the range bounded by the green lines.")
        print("The suggested cutoff for the Fourier filter is listed below,")
        for crossing in sign_crossing:
            print(f"{r[crossing]:.2f} Å")
        print("============================================================")
        r_cut = float(
            input("Input the cutoff for the Fourier filter: ")
        )
        plt.close()
        print(f"Using {r_cut:.2f} Å as the cutoff for the Fourier filter.")
        print("============================================================\n")

    # Fill in 0. as the starting point for the r-chunks, saving input efforts
    # for the user. Also, internally, we are replacing the last value in the 
    # r-chunk to the rmax value obtained based on the qbin value, according to
    # the Nyquist-Shannon theorem.
    r_chunks = np.insert(rchunks, 0, 0.)
    r_chunks[-1] = rmax
    sq_sum = np.zeros_like(q)

    info_msg = "[Info] Processing the data in "
    info_msg += f"{len(r_chunks) - 1} chunks ...\n"
    if not interactive:
        info_msg = "\n" + info_msg
    print(info_msg)

    if diagnostic:
        sq_chunks_all = list()
        r_chunks_all = list()
        gr_chunks_all = list()

    for i, q_chunk in enumerate(qchunks):
        q2, sq2, _ = transformer.apply_cropping(
            q, sq, qmin, q_chunk
        )
        sq2 = sq2 - np.mean(sq2[len(sq2) - int(5. / qbin): len(sq2)]) + 1.
        r, gr, _ = transformer.S_to_g(q2, sq2, r, rho=num_density)

        # The Fourier filtered Q-space data will be used for the next chunk.
        ff_out = ff.g_using_S(r, gr, q2, sq2, r_cut, rho=num_density)
        q = ff_out[2]
        sq = ff_out[3]
        r = ff_out[4]
        gr = ff_out[5]

        # The first chunk Fourier transform data is using the largest Qmax and
        # the data before cropping would be the real space data that would be
        # obtained through a direct Fourier transform without the
        # chunk-by-chunk processing.
        if i == 0:
            r_init = np.copy(r)
            gr_init = np.copy(gr)

        r_crop, gr_crop, _ = transformer.apply_cropping(
            r, gr, r_chunks[i], r_chunks[i + 1]
        )

        # Back Fourier transform the data to get the S(Q) data. This is just
        # like separating the overall integration range into several continuous
        # ranges and then summing them up. If without the different Qmax being
        # used in previous step for the Fourier transform into real space, the
        # data obtained here would be the same as the one obtained through a
        # direct Fourier transform over the same overall Q-range.
        q_crop = np.linspace(qbin, qmax_init, int(qmax_init / qbin))
        q_crop, sq_crop, _ = transformer.g_to_S(
            r_crop, gr_crop, q_crop, rho=num_density
        )

        sq_sum = np.add(sq_sum, sq_crop)

        print(
            f"[Info] Done with chunk {i + 1}, rmin = {r_chunks[i]:.2f} Å, "
            f"rmax = {r_chunks[i + 1]:.2f} Å"
        )

        if diagnostic:
            sq_chunks_all.append(sq_crop)
            r_chunks_all.append(r_crop)
            gr_chunks_all.append(gr_crop)

    sq_sum -= (len(qchunks) - 1.)

    # The final back Fourier transform to get the final g(r) data from the
    # summed S(Q) data.
    r_final = np.linspace(0., rmax, rmax_idx)
    r_final, gr_final, _ = transformer.S_to_g(
        q_crop, sq_sum, r_final, rho=num_density
    )

    # Convert S(Q) into specified output form.
    if q_out_form == "S(Q)":
        sq_out = sq_sum
        sq_init_out = sq_init
        qf_name = "sofq"
        qf_unit = ""
    elif q_out_form == "F(Q)":
        sq_out, _ = converter.S_to_F(q_crop, sq_sum)
        sq_init_out, _ = converter.S_to_F(q_init, sq_init)
        qf_name = "fofq"
        qf_unit = r"Å$^{-1}$"
    elif q_out_form == "FK(Q)":
        kwargs = {'<b_coh>^2': fzcoeff}
        sq_out, _ = converter.S_to_FK(q_crop, sq_sum, **kwargs)
        sq_init_out, _ = converter.S_to_FK(q_init, sq_init, **kwargs)
        qf_name = "fkofq"
        qf_unit = "barn"
    else:
        print(
            "[Warning] Unknown output form for Q-space output. "
            "Using 'S(Q)' as default."
        )
        sq_out = sq_sum
        sq_init_out = sq_init
        qf_name = "sofq"
        qf_unit = ""

    # Convert g(r) into specified output form.
    if r_out_form == "g(r)":
        gr_out = gr_final
        gr_init_out = gr_init
        rf_name = "gofr"
        rf_unit = ""
    elif r_out_form == "G(r)":
        kwargs = {'rho': num_density}
        gr_out, _ = converter.g_to_G(r_final, gr_final, **kwargs)
        gr_init_out, _ = converter.g_to_G(r_init, gr_init, **kwargs)
        rf_name = "pdf"
        rf_unit = r"Å$^{-2}$"
    elif r_out_form == "GK(r)":
        kwargs = {
            '<b_coh>^2': fzcoeff,
            'rho': num_density
        }
        gr_out, _ = converter.g_to_GK(r_final, gr_final, **kwargs)
        gr_init_out, _ = converter.g_to_GK(r_init, gr_init, **kwargs)
        rf_name = "gkofr"
        rf_unit = "barn"
    else:
        print(
            "[Warning] Unknown output form for r-space output. "
            "Using 'g(r)' as default."
        )
        gr_out = gr_final
        gr_init_out = gr_init
        rf_name = "gofr"
        rf_unit = ""

    r_out_crop, gr_out_crop, _ = transformer.apply_cropping(
        r_final, gr_out, rbin, rmax_out
    )
    r_init_crop, gr_init_crop, _ = transformer.apply_cropping(
        r_init, gr_init_out, rbin, rmax_out
    )

    # Output the final data in both Q- and r-space to files.
    out_sq = os.path.join(
        os.getcwd(),
        f"{output}_cbyc_ff_{qf_name}.sq"
    )
    out_gr = os.path.join(
        os.getcwd(),
        f"{output}_cbyc_ff_{rf_name}.gr"
    )

    rchunks_out = [float("{0:.2F}".format(rv)) for rv in rchunks]
    with open(out_sq, "w") as f:
        f.write(f"{len(q_crop)}\n")
        f.write(
            f"# QChunks: {qchunks}, "
            f"RChunks: {rchunks_out}\n"
        )
        for i in range(len(q_crop)):
            f.write(f"{q_crop[i]:10.3f}{sq_out[i]:15.6f}\n")

    with open(out_gr, "w") as f:
        f.write(f"{len(r_out_crop)}\n")
        f.write(
            f"# QChunks: {qchunks}, "
            f"RChunks: {rchunks_out}\n"
        )
        for i in range(len(r_out_crop)):
            f.write(f"{r_out_crop[i]:10.3f}{gr_out_crop[i]:15.6f}\n")

    # If the user wants to see the partition of the data, we will output them.
    if diagnostic:
        out_sq_diagnostic = os.path.join(
            os.getcwd(),
            f"{output}_cbyc_sofq_parts.sq"
        )
        out_gr_diagnostic = os.path.join(
            os.getcwd(),
            f"{output}_cbyc_gofr_parts.gr"
        )

        with open(out_sq_diagnostic, "w") as f:
            f.write(f"{len(q_crop)}\n")
            f.write(
                f"# QChunks: {qchunks}, "
                f"RChunks: {rchunks_out}\n"
            )
            for i in range(len(q_crop)):
                f.write(f"{q_crop[i]:10.3f}")
                for j in range(len(sq_chunks_all)):
                    f.write(f"{sq_chunks_all[j][i]:15.6f}")
                f.write("\n")

        with open(out_gr_diagnostic, "w") as f:
            f.write(f"{len(r_final)}\n")
            f.write(
                f"# QChunks: {qchunks}, "
                f"RChunks: {rchunks_out}\n"
            )
            for i in range(len(r_final)):
                f.write(f"{r_final[i]:10.3f}")
                for j in range(len(gr_chunks_all)):
                    indices = np.where(r_chunks_all[j] == r_final[i])[0]
                    if len(indices) > 0:
                        val_tmp = gr_chunks_all[j][indices[0]]
                    else:
                        val_tmp = 0.
                    f.write(f"{val_tmp:15.6f}")
                f.write("\n")

        labels = list()
        for i in range(len(r_chunks) - 1):
            labels.append(
                f"{r_chunks[i]:.2F} - {r_chunks[i + 1]:.2F}"
            )
        plot_handle.update_data(
            r_chunks_all,
            gr_chunks_all,
            labels,
            color_list[: len(r_chunks_all)],
            'r',
            'g(r)',
            'Å',
            '',
            title="Partition of the real space data"
        )
        fig, ax = plot_handle.plot()
        if interactive:
            plt.show()
        plt.close()

        max_r = max(max(rlist) for rlist in gr_chunks_all)
        ax.set_xlim(0., 15.)
        ax.annotate(
            "High r region not shown for clarity",
            xy=(7.5, max_r - 0.5),
            ha="center",
            fontsize=18,
            color=color_list[len(r_chunks_all) - 1]
        )
        fig.savefig(f"{output}_cbyc_gofr_parts.png", dpi=300)

        sq_chunks_all_f = list()
        offset = 0.
        for i in range(len(sq_chunks_all)):
            if i == 0:
                sq_chunks_all_f.append(sq_chunks_all[i])
            else:
                v_tmp = max(sq_chunks_all[i - 1]) - min(sq_chunks_all[i]) + .2
                offset += (v_tmp + .2)
                sq_chunks_all_f.append(sq_chunks_all[i] + offset)
        plot_handle.update_data(
            [q_crop for _ in range(len(sq_chunks_all))],
            sq_chunks_all_f,
            labels,
            color_list[: len(r_chunks_all)],
            'Q',
            'S(Q)',
            r'Å$^{-1}$',
            '',
            title="Fourier transform of the real space data partitions"
        )
        fig, ax = plot_handle.plot()
        ax.legend(loc='upper right', fontsize=18)
        if interactive:
            plt.show()
        plt.close()

        fig.savefig(f"{output}_cbyc_sofq_parts.png", dpi=300)

    if "K" in q_out_form:
        q_left_m = 0.12

    plot_handle.update_data(
        [q_init, q_crop],
        [sq_init_out, sq_out],
        ["Initial", "Fourier Filtered"],
        ["black", "red"],
        'Q',
        f'{q_out_form}',
        r'Å$^{-1}$',
        qf_unit,
        title=f"Comparison of {q_out_form} before and after the processing",
        left_m=q_left_m
    )
    fig, _ = plot_handle.plot()
    fig.savefig(f"{output}_cbyc_ff_{qf_name}.png", dpi=300)
    if interactive:
        plt.show()
    plt.close()

    if "K" in r_out_form:
        r_left_m = 0.12

    plot_handle.update_data(
        [r_init_crop, r_out_crop],
        [gr_init_crop, gr_out_crop],
        ["Initial", "Fourier Filtered"],
        ["black", "red"],
        'r',
        f'{r_out_form}',
        'Å',
        rf_unit,
        title=f"Comparison of {r_out_form} before and after the processing",
        left_m=r_left_m
    )
    fig, _ = plot_handle.plot()
    fig.savefig(f"{output}_cbyc_ff_{rf_name}.png", dpi=300)
    if interactive:
        plt.show()
    plt.close()

    print("\n============================================================")
    print("[Info] Output files of the processed data,")
    print(f"[Info] {out_sq}")
    print(f"[Info] {out_gr}")
    print("============================================================\n")
