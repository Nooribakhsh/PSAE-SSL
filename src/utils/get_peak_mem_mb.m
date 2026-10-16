function [mb, kind] = get_peak_mem_mb()
% GET_PEAK_MEM_MB  Peak resident memory (MB) of the CURRENT MATLAB process
%   since it started (OS high-water mark, not just the size of variables).
%   Linux  : VmHWM from /proc/self/status           (true peak RSS)
%   Windows: PeakWorkingSet64 via PowerShell        (true peak)
%   macOS  : current RSS via ps                     (NOT a peak; kind says so)
mb = NaN; kind = 'unknown';
try
    if isunix && ~ismac
        txt = fileread('/proc/self/status');
        tok = regexp(txt, 'VmHWM:\s+(\d+)\s+kB', 'tokens', 'once');
        mb = str2double(tok{1}) / 1024; kind = 'peak_rss';
    elseif ispc
        pid = get_pid();
        cmd = sprintf('powershell -NoProfile -Command "(Get-Process -Id %d).PeakWorkingSet64"', pid);
        [st, out] = system(cmd);
        if st == 0, mb = str2double(strtrim(out)) / 1048576; kind = 'peak_working_set'; end
    elseif ismac
        pid = get_pid();
        [st, out] = system(sprintf('ps -o rss= -p %d', pid));
        if st == 0, mb = str2double(strtrim(out)) / 1024; kind = 'current_rss_not_peak'; end
    end
catch
end
end

function pid = get_pid()
try
    pid = feature('getpid');
catch
    nm = char(java.lang.management.ManagementFactory.getRuntimeMXBean().getName());
    pid = str2double(strtok(nm, '@'));
end
end
