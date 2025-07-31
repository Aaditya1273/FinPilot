document.addEventListener('DOMContentLoaded', () => {

    // Loading screen
    const loadingScreen = document.querySelector('.loading');
    window.addEventListener('load', () => {
        loadingScreen.classList.add('hidden');
    });

    // Interactive orbs
    const orbs = document.querySelectorAll('.bg-orb');
    document.addEventListener('mousemove', (e) => {
        orbs.forEach(orb => {
            const speed = orb.getAttribute('data-speed') || 1;
            const x = (window.innerWidth - e.pageX * speed) / 100;
            const y = (window.innerHeight - e.pageY * speed) / 100;
            orb.style.transform = `translateX(${x}px) translateY(${y}px)`;
        });
    });

    // Form submission
    const emailForm = document.querySelector('.email-form');
    emailForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const emailInput = emailForm.querySelector('input[type="email"]');
        const submitButton = emailForm.querySelector('.email-submit');

        if (emailInput.value) {
            submitButton.textContent = 'Thanks!';
            submitButton.style.backgroundColor = '#26de81';
            setTimeout(() => {
                submitButton.textContent = 'Get Early Access';
                submitButton.style.backgroundColor = '';
                emailInput.value = '';
            }, 2000);
        } else {
            emailInput.focus();
        }
    });

    // Animate on scroll / intersection observer
    const animatedElements = document.querySelectorAll('[data-animation]');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('in-view');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    animatedElements.forEach(el => {
        observer.observe(el);
    });
});
