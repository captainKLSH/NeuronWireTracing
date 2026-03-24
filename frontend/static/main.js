
function setLabel(zoneId, labelId, input) {
    const label = document.getElementById(labelId);
    const zone = document.getElementById(zoneId);
    
    if (input.files && input.files.length > 0) {
        // 1. Get the filename
        const fileName = input.files[0].name;
        
        // 2. Update the text
        label.innerText = `📄 ${fileName}`;
        label.style.color = "#4ade80"; // Botanist Green
        
        // 3. Add a class to the zone to change its appearance
        zone.classList.add('file-uploaded');
    } else {
        label.innerText = "no file selected";
        zone.classList.remove('file-uploaded');
    }
}

// Animate bars on load
window.addEventListener('load', () => {
    document.querySelectorAll('.bar-fill').forEach(bar => {
        const target = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => { bar.style.width = target; }, 300);
    });
});
