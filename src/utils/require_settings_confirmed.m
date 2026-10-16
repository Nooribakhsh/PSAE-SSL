function require_settings_confirmed()
% REQUIRE_SETTINGS_CONFIRMED  Abort unless the user has reviewed settings.
usr = get_user_settings();
if ~isfield(usr, 'settingsConfirmed') || ~strcmpi(strtrim(usr.settingsConfirmed), 'yes')
    error(['Review src/utils/get_user_settings.m, set ' ...
           'usr.settingsConfirmed = ''yes'', and rerun.']);
end
end
