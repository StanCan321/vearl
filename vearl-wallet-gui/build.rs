// Windows: the program icon (the Vearl mark) and the version information shown in the file properties.
fn main() {
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/vearl.ico");
        res.set("ProductName", "Vearl Wallet");
        res.set("FileDescription", "Vearl Wallet");
        res.set("CompanyName", "Vearl");
        res.set("LegalCopyright", "Apache-2.0");
        res.set("OriginalFilename", "vearl-wallet-gui.exe");
        if let Err(e) = res.compile() {
            // a build without the resource compiler still works: only the icon in the file explorer is missing
            println!("cargo:warning=could not embed the icon: {e}");
        }
    }
}
