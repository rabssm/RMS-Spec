""" Batch run the flux code using a flux batch file. """

import datetime
import os
import shlex
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
from RMS.Astrometry.Conversions import datetime2JD, jd2Date
from RMS.Formats.FTPdetectinfo import findFTPdetectinfoFile
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert, solLon2jdSteyaert

from Utils.Flux import calculatePopulationIndex, computeFlux, detectClouds, fluxParser, loadForcedBinFluxData


def addFixedBins(sol_bins, small_sol_bins, *params):
    """
    For a larger array of solar longitudes sol_bins, fits parameters to an empty array of its size (minus 1)
    so that small_sol_bins agrees with sol_bins

    Assumes that for some index i, sol_bins[i:i+len(small_sol_bins)] = small_sol_bins. If this is not true,
    then the values are invalid and different small arrays should be used

    Arguments:
        sol_bins: [ndarray] Array of solar longitude bin edges. Does not wrap around
        small_sol_bins: [ndarray] Array of solar longitude bin edges which is smaller in length than
            sol_bins but can be transformed to sol_bins if shifted by a certain index. Does not wrap
            around.
        *params: [ndarray] Physical quantities such as number of meteors, collecting area.

    Return:
        [tuple] Same variables corresponding to params
            - val: [ndarray] Array of where any index that used to correspond to a sol in small_sol_bins,
                now corresponds to an index in sol_bins, padding all other values with zeros
    """
    # if sol_bins wraps would wrap around but forced_bins_sol doesn't
    if sol_bins[0] > small_sol_bins[0]:
        i = np.argmax(sol_bins - (small_sol_bins[0] + 360) > -1e-7)
    else:
        i = np.argmax(sol_bins - small_sol_bins[0] > -1e-7)  # index where they are equal

    data_arrays = []
    for p in params:
        forced_bin_param = np.zeros(len(sol_bins) - 1)
        forced_bin_param[i : i + len(p)] = p
        data_arrays.append(forced_bin_param)

    return data_arrays


def combineFixedBinsAndComputeFlux(
    sol_bins, meteors, time_area_prod, min_meteors=50, ci=0.95, min_tap=2, max_bin_duration=6
):
    """
    Computes flux values and their corresponding solar longitude based on bins containing
    number of meteors, and time-area product. Bins will be combined so that each bin has the
    minimum number of meteors

    Arguments:
        sol_bins: [ndarray] Solar longitude of bins start and end (the length must be 1 more than meteors)
        meteors: [ndarray] Number of meteors in a bin
        time_area_prod: [ndarray] Time multiplied by LM corrected collecting area added for each station
            which contains each bin
        time: [ndarray]

    Keyword arguments:
        min_meteors: [int] Minimum number of meteors to have in a bin
        ci: [float] Confidence interval for calculating the flux error bars (from 0 to 1)
        min_tap: [float] Minimum time area product in 1000 km^2*h.
        max_bin_duration: [float] Maximum bin duration in hours.

    Return:
        [tuple] sol, flux, flux_lower, flux_upper, meteors, ta_prod
            - sol: [ndarray] Solar longitude
            - flux: [ndarray] Flux corresponding to solar longitude
            - flux_lower: [ndarray] Lower bound of flux corresponding to sol
            - flux_upper: [ndarray] Upper bound of flux corresponding to sol
            - meteor_count: [ndarray] Number of meteors in bin
            - time_area_product: [ndarray] Time area product of bin

    """
    middle_bin_sol = (sol_bins[1:] + sol_bins[:-1]) / 2

    flux_list = []
    flux_upper_list = []
    flux_lower_list = []
    sol_list = []
    sol_bin_list = []
    meteor_count_list = []
    area_time_product_list = []

    start_idx = 0
    for end_idx in range(1, len(meteors)):
        sl = slice(start_idx, end_idx)
        if np.sum(meteors[sl]) >= min_meteors and np.nansum(time_area_prod[sl]) / 1e9 >= min_tap:
            ta_prod = np.sum(time_area_prod[sl])

            num_meteors = np.sum(meteors[sl])
            meteor_count_list.append(num_meteors)
            area_time_product_list.append(ta_prod)
            if ta_prod == 0:
                flux_list.append(0)
                flux_upper_list.append(0)
                flux_lower_list.append(0)
            else:
                n_meteors_upper = scipy.stats.chi2.ppf(0.5 + ci / 2, 2 * (num_meteors + 1)) / 2
                n_meteors_lower = scipy.stats.chi2.ppf(0.5 - ci / 2, 2 * num_meteors) / 2
                flux_list.append(1e9 * num_meteors / ta_prod)
                flux_upper_list.append(1e9 * n_meteors_upper / ta_prod)
                flux_lower_list.append(1e9 * n_meteors_lower / ta_prod)

            sol_list.append(np.mean(middle_bin_sol[sl]))
            sol_bin_list.append(sol_bins[start_idx])
            start_idx = end_idx
        elif (middle_bin_sol[end_idx] - middle_bin_sol[start_idx]) / (
            2 * np.pi
        ) * 24 * 365.24219 >= max_bin_duration:
            start_idx = end_idx

    sol_bin_list.append(sol_bins[start_idx])

    return (
        np.array(sol_list),
        np.array(sol_bin_list),
        np.array(flux_list),
        np.array(flux_lower_list),
        np.array(flux_upper_list),
        np.array(meteor_count_list),
        np.array(area_time_product_list),
    )


def calculateFixedBins(all_time_intervals, dir_list, bin_duration=5):
    """
    Function to calculate the bins that any amount of stations over any number of years for one shower
    can be put into.

    Arguments:
        file_data: [list]

    Keyword arguments:
        bin_duration: [float] Bin duration in minutes (this is only an approximation since the bins are
            fixed to solar longitude)

    Return:
        [tuple] sol_bins, bin_datetime_dict
            - sol_bins: [ndarray] array of solar longitudes corresponding to each of the bins
            - bin_datetime_dict: [list] Each element contains a list of two elements: datetime_range, bin_datetime
                - datetime_range: [tuple] beg_time, end_time
                    - beg_time: [datetime] starting time of the bins
                    - end_time: [datetime] ending time of the last bin
                - bin_datetime: [list] list of datetime for each bin start and end time. The first element
                    is beg_time and the last element is end_time
    """

    # calculate bins for summary calculations
    if not all_time_intervals:
        return np.array([]), []

    sol_delta = 2 * np.pi / 60 / 24 / 365.24219 * bin_duration
    sol_beg = np.array([jd2SolLonSteyaert(datetime2JD(beg)) for beg, _ in all_time_intervals])
    sol_end = np.array([jd2SolLonSteyaert(datetime2JD(end)) for _, end in all_time_intervals])

    # even if the solar longitude wrapped around, make sure that you know what the smallest sol are
    if np.max(sol_beg) - np.min(sol_beg) > np.pi or np.max(sol_end) - np.min(sol_end) > np.pi:
        start_idx = np.argmin(np.where(sol_beg > np.pi, sol_beg, 2 * np.pi))
        end_idx = np.argmax(np.where(sol_end <= np.pi, sol_beg, 0))
    else:
        start_idx = np.argmin(sol_beg)
        end_idx = np.argmax(sol_end)
    min_sol = sol_beg[start_idx]
    max_sol = sol_end[end_idx] if sol_beg[start_idx] < sol_end[end_idx] else sol_end[end_idx] + 2 * np.pi
    sol_bins = np.arange(min_sol, max_sol, sol_delta)
    sol_bins = np.append(sol_bins, sol_bins[-1] + sol_delta)  # all events should be within the bins

    # make sure that fixed bins fit with already existing bins saved
    existing_sol = []
    for _dir in dir_list:
        loaded_sol = []
        for filename in os.listdir(_dir):
            if 'fixedbinsflux' in filename and filename.endswith('.csv'):
                loaded_sol.append(loadForcedBinFluxData(_dir, filename)[0])
        if loaded_sol:
            existing_sol.append(loaded_sol)
        # does not check to make sure that none of the intervals during a single night overlap. User must
        # make sure of this

    ## calculating sol_bins
    if existing_sol:
        # select a starting_sol to transform sol_bins to so that it matched what already exists
        if len(existing_sol) == 1:
            starting_sol = existing_sol[0][0]
        else:
            # if there's more than one array of sol values, make sure they all agree with each other and
            # take the first
            failed = False
            comparison_sol = existing_sol[0][0]
            for loaded_sol_list, _dir in zip(existing_sol[1:], dir_list[1:]):
                sol = loaded_sol_list[0]  # take the first of the list and assume the others agree with it
                min_len = min(len(comparison_sol), len(sol), 5)
                a, b = (
                    (comparison_sol[:min_len], sol[:min_len])
                    if comparison_sol[0] > sol[0]
                    else (sol[:min_len], comparison_sol[:min_len])
                )
                epsilon = 1e-7
                goal = sol_delta / 2
                val = (np.median(a - b if a[0] - b[0] < np.pi else b + 2 * np.pi - a) + goal) % sol_delta
                if np.abs(goal - val) > epsilon:
                    print(
                        "!!! fixedbinsflux.csv in {:s} and {:s} don't match solar longitude values".format(dir_list[0], _dir)
                    )
                    print('\tSolar longitude difference:', np.abs(goal - val))
                    failed = True

            if failed:
                print()
                raise Exception(
                    'Flux bin solar longitudes didn\'t match. To fix this, at least one of the'
                    ' fixedbinsflux.csv must be deleted.'
                )
            # filter only sol values that are inside the solar longitude
            starting_sol = comparison_sol

        # adjust bins to fit existing bins
        length = min(len(starting_sol), len(sol_bins))
        sol_bins += np.mean(starting_sol[:length] - sol_bins[:length]) % sol_delta
        # make sure that sol_bins satisfies the range even with the fit
        sol_bins = np.append(sol_bins[0] - sol_delta, sol_bins)  # assume that it doesn't wrap around

    ## calculating datetime corresponding to sol_bins for each year
    bin_datetime_dict = []
    bin_datetime = []
    for sol in sol_bins:
        curr_time = all_time_intervals[start_idx][0] + datetime.timedelta(
            minutes=(sol - sol_bins[0]) / (2 * np.pi) * 365.24219 * 24 * 60
        )
        bin_datetime.append(jd2Date(solLon2jdSteyaert(curr_time.year, curr_time.month, sol), dt_obj=True))
    bin_datetime_dict.append([(bin_datetime[0], bin_datetime[-1]), bin_datetime])

    for start_time, _ in all_time_intervals:
        if all(
            [
                year_start > start_time or start_time > year_end
                for (year_start, year_end), _ in bin_datetime_dict
            ]
        ):
            delta_years = int(
                np.floor(
                    (start_time - all_time_intervals[start_idx][0]).total_seconds()
                    / (365.24219 * 24 * 60 * 60)
                )
            )
            bin_datetime = [
                jd2Date(solLon2jdSteyaert(dt.year + delta_years, dt.month, sol), dt_obj=True)
                for sol, dt in zip(sol_bins, bin_datetime_dict[0][1])
            ]
            bin_datetime_dict.append([(bin_datetime[0], bin_datetime[-1]), bin_datetime])

    return sol_bins, bin_datetime_dict


class StationPlotParams:
    '''Class to give plots specific appearances based on the station'''

    def __init__(self):
        self.color_dict = {}
        self.marker_dict = {}
        self.markers = ['o', 'x', '+']

        self.color_cycle = [plt.get_cmap("tab10")(i) for i in range(10)]

    def __call__(self, station):
        if station not in self.color_dict:
            # Generate a new color
            color = self.color_cycle[len(self.color_dict) % (len(self.color_cycle))]
            label = station
            marker = self.markers[(len(self.marker_dict) // 10) % (len(self.markers))]

            # Assign plot color
            self.color_dict[station] = color
            self.marker_dict[station] = marker

        else:
            color = self.color_dict[station]
            marker = self.marker_dict[station]
            # label = str(config.stationID)
            label = None

        return {'color': color, 'marker': marker, 'label': label}


if __name__ == "__main__":

    import argparse

    import RMS.ConfigReader as cr

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(
        description="Compute multi-station and multi-year meteor shower flux from a batch file."
    )

    arg_parser.add_argument("batch_path", metavar="BATCH_PATH", type=str, help="Path to the flux batch file.")
    arg_parser.add_argument(
        "--output_filename",
        metavar="FILENAME",
        type=str,
        default='fluxbatch_output',
        help="Filename to export images and data (exclude file extensions), defaults to fluxbatch_output",
    )
    arg_parser.add_argument(
        "-csv",
        action='store_true',
        help="If given, will read from the csv files defined with output_filename (defaults to fluxbatch_output)",
    )
    arg_parser.add_argument(
        "--minmeteors",
        type=int,
        default=30,
        help="Minimum meteors per bin. If this is not satisfied the bin will be made larger",
    )
    arg_parser.add_argument(
        "--mintap",
        type=float,
        default=3,
        help="Minimum time-area product per bin. If this is not satisfied the bin will be made larger",
    )
    arg_parser.add_argument(
        "--maxduration",
        type=float,
        default=6,
        help="Maximum time per bin in hours. If this is not satisfied, the bin will be discarded.",
    )

    # Parse the command line arguments
    fluxbatch_cml_args = arg_parser.parse_args()

    #########################


    ### Binning parameters ###

    # Confidence interval
    ci = 0.95

    # Minimum bin duration (minutes)
    bin_duration = 5

    # Minimum number of meteors in the bin
    min_meteors = fluxbatch_cml_args.minmeteors

    # Minimum time-area product (1000 km^2 h)
    min_tap = fluxbatch_cml_args.mintap

    # Maximum bin duration
    max_bin_duration = fluxbatch_cml_args.maxduration

    ### ###


    # Check if the batch file exists
    if not os.path.isfile(fluxbatch_cml_args.batch_path):
        print("The given batch file does not exist!", fluxbatch_cml_args.batch_path)
        sys.exit()

    dir_path = os.path.dirname(fluxbatch_cml_args.batch_path)

    output_data = []
    shower_code = None
    summary_population_index = []

    plot_info = StationPlotParams()

    fig, ax = plt.subplots(2, figsize=(15, 8), sharex=True)


    if not fluxbatch_cml_args.csv:

        # loading commands from batch file and collecting information to run computeflux, including
        # detecting the clouds

        file_data = []
        with open(fluxbatch_cml_args.batch_path) as f:

            # Parse the batch entries
            for line in f:
                line = line.replace("\n", "").replace("\r", "")

                if not len(line):
                    continue

                if line.startswith("#"):
                    continue

                flux_cml_args = fluxParser().parse_args(shlex.split(line))
                (
                    ftpdetectinfo_path,
                    shower_code,
                    s,
                    binduration,
                    binmeteors,
                    time_intervals,
                    fwhm,
                    ratio_threshold,
                ) = (
                    flux_cml_args.ftpdetectinfo_path,
                    flux_cml_args.shower_code,
                    flux_cml_args.s,
                    flux_cml_args.binduration,
                    flux_cml_args.binmeteors,
                    flux_cml_args.timeinterval,
                    flux_cml_args.fwhm,
                    flux_cml_args.ratiothres,
                )
                ftpdetectinfo_path = findFTPdetectinfoFile(ftpdetectinfo_path)

                if not os.path.isfile(ftpdetectinfo_path):
                    print("The FTPdetectinfo file does not exist:", ftpdetectinfo_path)
                    print("Exiting...")
                    sys.exit()

                # Extract parent directory
                ftp_dir_path = os.path.dirname(ftpdetectinfo_path)

                # Load the config file
                config = cr.loadConfigFromDirectory('.', ftp_dir_path)
                if time_intervals is None:
                    # find time intervals to compute flux with
                    print('Detecting whether clouds are present...')
                    time_intervals = detectClouds(
                        config, ftp_dir_path, show_plots=False, ratio_threshold=ratio_threshold
                    )
                    print('Cloud detection complete!')
                    print()
                else:
                    time_intervals = [(*time_intervals,)]

                file_data.append(
                    [
                        config,
                        ftp_dir_path,
                        ftpdetectinfo_path,
                        shower_code,
                        time_intervals,
                        s,
                        binduration,
                        binmeteors,
                        fwhm,
                    ]
                )

        sol_bins, bin_datetime_dict = calculateFixedBins(
            [time_interval for data in file_data for time_interval in data[4]],
            [data[1] for data in file_data],
            bin_duration=bin_duration,
        )

        all_bin_information = []
        # Compute the flux
        for (
            config,
            ftp_dir_path,
            ftpdetectinfo_path,
            shower_code,
            time_intervals,
            s,
            binduration,
            binmeteors,
            fwhm,
        ) in file_data:
            for interval in time_intervals:
                dt_beg, dt_end = interval
                forced_bins = (
                    bin_datetime_dict[
                        np.argmax(
                            [
                                year_start < dt_beg < year_end
                                for (year_start, year_end), _ in bin_datetime_dict
                            ]
                        )
                    ][1],
                    sol_bins,
                )
                ret = computeFlux(
                    config,
                    ftp_dir_path,
                    ftpdetectinfo_path,
                    shower_code,
                    dt_beg,
                    dt_end,
                    s,
                    binduration,
                    binmeteors,
                    show_plots=False,
                    default_fwhm=fwhm,
                    forced_bins=forced_bins,
                    confidence_interval=ci,
                )
                if ret is None:
                    continue
                (
                    sol_data,
                    flux_lm_6_5_data,
                    flux_lm_6_5_ci_lower_data,
                    flux_lm_6_5_ci_upper_data,
                    meteor_num_data,
                    population_index,
                    bin_information,
                ) = ret

                # Add computed flux to the output list
                summary_population_index.append(population_index)
                output_data += [
                    [config.stationID, sol, flux, lower, upper, population_index]
                    for (sol, flux, lower, upper) in zip(
                        sol_data, flux_lm_6_5_data, flux_lm_6_5_ci_lower_data, flux_lm_6_5_ci_upper_data
                    )
                ]

                all_bin_information.append(addFixedBins(sol_bins, *bin_information))

                # plot data for night and interval
                plot_params = plot_info(config.stationID)
                line = ax[0].plot(sol_data, flux_lm_6_5_data, **plot_params, linestyle='dashed')

                ax[0].errorbar(
                    sol_data,
                    flux_lm_6_5_data,
                    color=plot_params['color'],
                    alpha=0.5,
                    capsize=5,
                    zorder=3,
                    linestyle='none',
                    yerr=[
                        np.array(flux_lm_6_5_data) - np.array(flux_lm_6_5_ci_lower_data),
                        np.array(flux_lm_6_5_ci_upper_data) - np.array(flux_lm_6_5_data),
                    ],
                )

        # plotting data
        num_meteors = sum(np.array(meteors) for meteors, _, _ in all_bin_information)

        area_time_product = sum(np.array(area) * np.array(time) for _, area, time in all_bin_information)
        (
            comb_sol,
            comb_sol_bins,
            comb_flux,
            comb_flux_lower,
            comb_flux_upper,
            comb_num_meteors,
            comb_ta_prod,
        ) = combineFixedBinsAndComputeFlux(
            sol_bins,
            num_meteors,
            area_time_product,
            ci=ci,
            min_tap=min_tap,
            min_meteors=min_meteors,
            max_bin_duration=max_bin_duration,
        )
        comb_sol = np.degrees(comb_sol)
        comb_sol_bins = np.degrees(comb_sol_bins)

    else:

        # get list of directories so that fixedfluxbin csv files can be found
        with open(fluxbatch_cml_args.batch_path) as f:
            # Parse the batch entries
            for line in f:
                line = line.replace("\n", "").replace("\r", "")

                if not len(line):
                    continue

                if line.startswith("#"):
                    continue

                flux_cml_args = fluxParser().parse_args(shlex.split(line))
                shower_code = flux_cml_args.shower_code
                summary_population_index.append(calculatePopulationIndex(flux_cml_args.s))

        # load data from .csv file and plot it
        dirname = os.path.dirname(fluxbatch_cml_args.batch_path)
        data1 = np.genfromtxt(
            os.path.join(dirname, fluxbatch_cml_args.output_filename + "_1.csv"),
            delimiter=',',
            dtype=None,
            encoding=None,
            skip_header=1,
        )

        station_list = []
        for stationID, sol, flux, lower, upper, _ in data1:
            plot_params = plot_info(stationID)

            ax[0].errorbar(
                sol,
                flux,
                **plot_params,
                alpha=0.5,
                capsize=5,
                zorder=3,
                linestyle='none',
                yerr=[[flux - lower], [upper - flux]],
            )

        if os.path.exists(os.path.join(dirname, fluxbatch_cml_args.output_filename + "_2.csv")):
            data2 = np.genfromtxt(
                os.path.join(dirname, fluxbatch_cml_args.output_filename + "_2.csv"),
                delimiter=',',
                encoding=None,
                skip_header=1,
            )

            comb_sol_bins = data2[:, 0]
            comb_sol = data2[:-1, 1]
            comb_flux = data2[:-1, 2]
            comb_flux_lower = data2[:-1, 3]
            comb_flux_upper = data2[:-1, 4]
            comb_ta_prod = data2[:-1, 5]
            comb_num_meteors = data2[:-1, 6]
        else:
            comb_sol = []
            comb_sol_bins = []
            comb_flux = []
            comb_flux_lower = []
            comb_flux_upper = []
            comb_num_meteors = []
            comb_ta_prod = []

    if len(comb_sol):
        # plotting 5 minute bin data
        ax[0].errorbar(
            comb_sol % 360,
            comb_flux,
            yerr=[comb_flux - comb_flux_lower, comb_flux_upper - comb_flux],
            label='weighted average flux',
            c='k',
            marker='o',
            linestyle='none',
        )

        plot1 = ax[1].bar(
            ((comb_sol_bins[1:] + comb_sol_bins[:-1]) / 2) % 360,
            comb_ta_prod / 1e9,
            comb_sol_bins[1:] - comb_sol_bins[:-1],
            label='Time-area product',
        )

        ax[1].hlines(
            min_tap,
            np.min(comb_sol%360),
            np.max(comb_sol%360),
            colors='b',
            linestyles='--',
        )
        side_ax = ax[1].twinx()
        plot2 = side_ax.scatter(comb_sol%360, comb_num_meteors, c='k', label='Num meteors')
        side_ax.hlines(
            min_meteors,
            np.min(comb_sol%360),
            np.max(comb_sol%360),
            colors='k',
            linestyles='--',
        )
        side_ax.set_ylabel('Num meteors')
        side_ax.set_ylim(bottom=0)
        ax[1].legend([plot1, plot2], [plot1.get_label(), plot2.get_label()])

    # Show plot
    ax[0].legend()
    ax[0].set_title('{:s} r = {:.3f}'.format(shower_code, np.mean(summary_population_index)))
    ax[0].set_ylabel("Flux (meteoroids / 1000km$^2$ h)")
    ax[1].set_ylabel("Time-area product +6.5M (1000 km$^2$ h)")
    ax[1].set_xlabel("Solar longitude (deg)")
    # plt.tight_layout()

    fig_path = os.path.join(dir_path, fluxbatch_cml_args.output_filename + ".png")
    print("Figure saved to:", fig_path)
    plt.savefig(fig_path, dpi=300)

    plt.show()

    if not fluxbatch_cml_args.csv:
        if len(comb_sol):
            data_out_path = os.path.join(dir_path, fluxbatch_cml_args.output_filename + "_2.csv")
            with open(data_out_path, 'w') as fout:
                fout.write(
                    "# Sol bin start (deg), Mean Sol (deg), Flux@+6.5M (met/1000km^2/h), Flux lower bound, Flux upper bound, Area-time product (corrected to +6.5M) (m^3/h), Meteor Count\n"
                )
                for _sol_bins, _sol, _flux, _flux_lower, _flux_upper, _at, _nmeteors in zip(
                    comb_sol_bins,
                    comb_sol,
                    comb_flux,
                    comb_flux_lower,
                    comb_flux_upper,
                    comb_ta_prod,
                    comb_num_meteors,
                ):

                    fout.write(f"{_sol_bins},{_sol},{_flux},{_flux_lower},{_flux_upper},{_at},{_nmeteors}\n")
                fout.write(f"{comb_sol_bins[-1]},,,,,,\n")
            print("Data saved to:", data_out_path)

        data_out_path = os.path.join(dir_path, f"{fluxbatch_cml_args.output_filename}_1.csv")
        with open(data_out_path, 'w') as fout:
            fout.write(
                "# Station, Sol (deg), Flux@+6.5M (met/1000km^2/h), Flux lower bound, Flux upper bound, Population Index\n"
            )
            for entry in output_data:
                print(entry)
                stationID, sol, flux, lower, upper, population_index = entry

                fout.write(
                    "{:s},{:.8f},{:.3f},{:.3f},{:.3f},{}\n".format(
                        stationID, sol, flux, lower, upper, population_index
                    )
                )
        print("Data saved to:", data_out_path)
