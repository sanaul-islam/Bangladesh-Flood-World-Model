interface HeaderProps {
  backendOnline: boolean;
}

export function Header({
  backendOnline,
}: HeaderProps) {
  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="header-brand">
          <strong>
            BANGLADESH FLOOD RESPONSE
          </strong>

          <span>
            Forecast-aware evacuation decision support
          </span>
        </div>

        <div className="header-status">
          <span
            className={`status-dot ${
              backendOnline
                ? ""
                : "offline"
            }`}
          />

          <span>
            {backendOnline
              ? "SYSTEM ONLINE"
              : "SYSTEM OFFLINE"}
          </span>

          <span className="header-version">
            API
          </span>
        </div>
      </div>
    </header>
  );
}

export default Header;
